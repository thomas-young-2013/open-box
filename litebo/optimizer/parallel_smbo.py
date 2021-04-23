import sys
import time
import traceback
from typing import List
from collections import OrderedDict

from litebo.utils.constants import MAXINT, SUCCESS, FAILED, TIMEOUT
from litebo.core.computation.parallel_process import ParallelEvaluation
from multiprocessing import Lock
from litebo.utils.limit import time_limit, TimeoutException
from litebo.utils.util_funcs import get_result
from litebo.core.sync_batch_advisor import SyncBatchAdvisor
from litebo.core.async_batch_advisor import AsyncBatchAdvisor
from litebo.optimizer.base import BOBase


def wrapper(param):
    objective_function, config, time_limit_per_trial = param
    try:
        args, kwargs = (config,), dict()
        timeout_status, _result = time_limit(objective_function, time_limit_per_trial,
                                             args=args, kwargs=kwargs)
        if timeout_status:
            raise TimeoutException('Timeout: time limit for this evaluation is %.1fs' % time_limit_per_trial)
        else:
            objs, constraints = get_result(_result, FAILED_PERF=None)
    except Exception as e:
        if isinstance(e, TimeoutException):
            trial_state = TIMEOUT
        else:
            traceback.print_exc(file=sys.stdout)
            trial_state = FAILED
        objs = None
        constraints = None
    return [config, constraints, objs]


class pSMBO(BOBase):
    def __init__(self, objective_function, config_space,
                 parallel_strategy='async',
                 batch_size=4,
                 batch_strategy='median_imputation',
                 num_constraints=0,
                 num_objs=1,
                 sample_strategy: str = 'bo',
                 max_runs=200,
                 time_limit_per_trial=180,
                 surrogate_type=None,
                 acq_type=None,
                 acq_optimizer_type='local_random',
                 initial_runs=3,
                 init_strategy='random_explore_first',
                 initial_configurations=None,
                 ref_point=None,
                 history_bo_data: List[OrderedDict] = None,
                 logging_dir='logs',
                 task_id=None,
                 random_state=1,
                 ):

        if task_id is None:
            raise ValueError('Task id is not SPECIFIED. Please input task id first.')

        self.task_info = {'num_constraints': num_constraints, 'num_objs': num_objs}
        self.FAILED_PERF = [MAXINT] * num_objs
        super().__init__(objective_function, config_space, task_id=task_id, output_dir=logging_dir,
                         random_state=random_state, initial_runs=initial_runs, max_runs=max_runs,
                         sample_strategy=sample_strategy, time_limit_per_trial=time_limit_per_trial,
                         history_bo_data=history_bo_data)

        if parallel_strategy == 'sync':
            self.config_advisor = SyncBatchAdvisor(config_space, self.task_info,
                                                   batch_size=batch_size,
                                                   batch_strategy=batch_strategy,
                                                   initial_trials=initial_runs,
                                                   initial_configurations=initial_configurations,
                                                   init_strategy=init_strategy,
                                                   history_bo_data=history_bo_data,
                                                   optimization_strategy=sample_strategy,
                                                   surrogate_type=surrogate_type,
                                                   acq_type=acq_type,
                                                   acq_optimizer_type=acq_optimizer_type,
                                                   ref_point=ref_point,
                                                   task_id=task_id,
                                                   output_dir=logging_dir,
                                                   random_state=random_state)
        elif parallel_strategy == 'async':
            self.config_advisor = AsyncBatchAdvisor(config_space, self.task_info,
                                                    batch_size=batch_size,
                                                    batch_strategy=batch_strategy,
                                                    initial_trials=initial_runs,
                                                    initial_configurations=initial_configurations,
                                                    init_strategy=init_strategy,
                                                    history_bo_data=history_bo_data,
                                                    optimization_strategy=sample_strategy,
                                                    surrogate_type=surrogate_type,
                                                    acq_type=acq_type,
                                                    acq_optimizer_type=acq_optimizer_type,
                                                    ref_point=ref_point,
                                                    task_id=task_id,
                                                    output_dir=logging_dir,
                                                    random_state=random_state)
            self.update_lock = Lock()
        else:
            raise ValueError('Invalid parallel strategy - %s.' % parallel_strategy)

        self.parallel_strategy = parallel_strategy
        self.batch_size = batch_size

    def callback(self, result):
        _config, _constraints, _objs = result
        if _objs is None:
            _objs = self.FAILED_PERF
        _observation = [_config, SUCCESS, _constraints, _objs]
        # Report the result, and remove the config from the running queue.
        with self.update_lock:
            self.config_advisor.update_observation(_observation)
            self.logger.info('Update observation %d: %s.' % (self.iteration_id, str(_observation)))
        # Parent process: collect the result and increment id.
        self.iteration_id += 1

    def async_run(self):
        with ParallelEvaluation(wrapper, n_worker=self.batch_size) as proc:
            while self.iteration_id < self.max_iterations:
                _config = self.config_advisor.get_suggestion()
                _param = [self.objective_function, _config, self.time_limit_per_trial]
                # Submit a job to worker.
                proc.process_pool.apply_async(wrapper, (_param,), callback=self.callback)
                while len(self.config_advisor.running_configs) >= self.batch_size:
                    # Wait for workers.
                    time.sleep(0.3)

    # Asynchronously evaluate n configs
    def async_iterate(self, n=1):
        iter_id = 0
        with ParallelEvaluation(wrapper, n_worker=self.batch_size) as proc:
            while iter_id < n:
                _config = self.config_advisor.get_suggestion()
                _param = [self.objective_function, _config, self.time_limit_per_trial]
                # Submit a job to worker.
                proc.process_pool.apply_async(wrapper, (_param,), callback=self.callback)
                while len(self.config_advisor.running_configs) >= self.batch_size:
                    # Wait for workers.
                    time.sleep(0.3)
                iter_id += 1
        return self.config_advisor.total_configurations[-n:], self.config_advisor.total_state[
                                                              -n:], self.config_advisor.total_perfs[-n:]

    def sync_run(self):
        with ParallelEvaluation(wrapper, n_worker=self.batch_size) as proc:
            batch_num = (self.max_iterations + self.batch_size - 1) // self.batch_size
            if self.batch_size > self.config_advisor.init_num:
                batch_num += 1  # fix bug
            batch_id = 0
            while batch_id < batch_num:
                configs = self.config_advisor.get_suggestions()
                self.logger.info('Running on %d configs in the %d-th batch.' % (len(configs), batch_id))
                params = [(self.objective_function, config, self.time_limit_per_trial) for config in configs]
                # Wait all workers to complete their corresponding jobs.
                results = proc.parallel_execute(params)
                # Report their results.
                for idx, _result in enumerate(results):
                    _config, _constraints, _objs = _result
                    if _objs is None:
                        _objs = self.FAILED_PERF
                    _observation = [_config, SUCCESS, _constraints, _objs]
                    self.config_advisor.update_observation(_observation)
                    self.logger.info('In the %d-th batch [%d], using config %s, result is: %s'
                                     % (batch_id, idx, str(_config), str(_objs)))
                batch_id += 1

    def run(self):
        if self.parallel_strategy == 'async':
            self.async_run()
        else:
            self.sync_run()
        return self.get_history()
