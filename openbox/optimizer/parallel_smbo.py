import sys
import time
import traceback
from typing import List
from collections import OrderedDict

from openbox.utils.constants import MAXINT, SUCCESS, FAILED, TIMEOUT
from openbox.core.computation.parallel_process import ParallelEvaluation
from multiprocessing import Lock
from openbox.utils.limit import time_limit, TimeoutException
from openbox.utils.util_funcs import get_result
from openbox.core.sync_batch_advisor import SyncBatchAdvisor
from openbox.core.async_batch_advisor import AsyncBatchAdvisor
from openbox.core.ea_advisor import EA_Advisor
from openbox.core.base import Observation
from openbox.optimizer.base import BOBase


def wrapper(param):
    objective_function, config, time_limit_per_trial = param
    trial_state = SUCCESS
    start_time = time.time()
    try:
        args, kwargs = (config,), dict()
        timeout_status, _result = time_limit(objective_function, time_limit_per_trial,
                                             args=args, kwargs=kwargs)
        if timeout_status:
            raise TimeoutException('Timeout: time limit for this evaluation is %.1fs' % time_limit_per_trial)
        else:
            objs, constraints = get_result(_result)
    except Exception as e:
        if isinstance(e, TimeoutException):
            trial_state = TIMEOUT
        else:
            traceback.print_exc(file=sys.stdout)
            trial_state = FAILED
        objs = None
        constraints = None
    elapsed_time = time.time() - start_time
    return Observation(config, trial_state, constraints, objs, elapsed_time)


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
                 task_id='default_task_id',
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
            if sample_strategy in ['random', 'bo']:
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
            elif sample_strategy == 'ea':
                assert num_objs == 1 and num_constraints == 0
                self.config_advisor = EA_Advisor(config_space, self.task_info,
                                                 optimization_strategy=sample_strategy,
                                                 batch_size=batch_size,
                                                 task_id=task_id,
                                                 output_dir=logging_dir,
                                                 random_state=random_state)
            else:
                raise ValueError('Unknown sample_strategy: %s' % sample_strategy)
        elif parallel_strategy == 'async':
            self.advisor_lock = Lock()
            if sample_strategy in ['random', 'bo']:
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
            elif sample_strategy == 'ea':
                assert num_objs == 1 and num_constraints == 0
                self.config_advisor = EA_Advisor(config_space, self.task_info,
                                                 optimization_strategy=sample_strategy,
                                                 batch_size=batch_size,
                                                 task_id=task_id,
                                                 output_dir=logging_dir,
                                                 random_state=random_state)
            else:
                raise ValueError('Unknown sample_strategy: %s' % sample_strategy)
        else:
            raise ValueError('Invalid parallel strategy - %s.' % parallel_strategy)

        self.parallel_strategy = parallel_strategy
        self.batch_size = batch_size

    def callback(self, observation: Observation):
        config, trial_state, constraints, objs, elapsed_time = observation
        if objs is None:
            observation = Observation(config, trial_state, constraints, self.FAILED_PERF, elapsed_time)
        # Report the result, and remove the config from the running queue.
        with self.advisor_lock:
            # Parent process: collect the result and increment id.
            self.config_advisor.update_observation(observation)
            self.logger.info('Update observation %d: %s.' % (self.iteration_id + 1, str(observation)))
            self.iteration_id += 1  # must increment id after updating

    # TODO: Wrong logic. Need to wait before return?
    def async_run(self):
        with ParallelEvaluation(wrapper, n_worker=self.batch_size) as proc:
            while self.iteration_id < self.max_iterations:
                with self.advisor_lock:
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
        res_list = list()
        with ParallelEvaluation(wrapper, n_worker=self.batch_size) as proc:
            while iter_id < n:
                with self.advisor_lock:
                    _config = self.config_advisor.get_suggestion()
                _param = [self.objective_function, _config, self.time_limit_per_trial]
                # Submit a job to worker.
                res_list.append(proc.process_pool.apply_async(wrapper, (_param,), callback=self.callback))
                while len(self.config_advisor.running_configs) >= self.batch_size:
                    # Wait for workers.
                    time.sleep(0.3)
                iter_id += 1
            for res in res_list:
                res.wait()

        history = self.get_history()
        iter_config = history.configurations[-n:]
        iter_trial_state = history.trial_states[-n:]
        iter_constraints = history.constraint_perfs[-n:] if self.task_info['num_constraints'] > 0 else None
        iter_objs = history.perfs[-n:]     # caution: one dim if num_objs==1, different from SMBO.iterate()
        return iter_config, iter_trial_state, iter_constraints, iter_objs

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
                observations = proc.parallel_execute(params)
                # Report their results.
                for idx, observation in enumerate(observations):
                    config, trial_state, constraints, objs, elapsed_time = observation
                    if objs is None:
                        objs = self.FAILED_PERF
                        observation = Observation(config, trial_state, constraints, self.FAILED_PERF, elapsed_time)
                    self.config_advisor.update_observation(observation)
                    if self.task_info['num_constraints'] > 0:
                        self.logger.info('In the %d-th batch [%d/%d], config: %s, objective value: %s, constraints: %s.'
                                         % (batch_id, idx+1, len(configs), config, objs, constraints))
                    else:
                        self.logger.info('In the %d-th batch [%d/%d], config: %s, objective value: %s.'
                                         % (batch_id, idx+1, len(configs), config, objs))
                batch_id += 1

    def run(self):
        if self.parallel_strategy == 'async':
            self.async_run()
        else:
            self.sync_run()
        return self.get_history()
