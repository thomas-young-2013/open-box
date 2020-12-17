import sys
import time
import traceback
from typing import List
from collections import OrderedDict

from litebo.utils.constants import MAXINT, SUCCESS, FAILED, TIMEOUT
from litebo.core.computation.parallel_process import ParallelEvaluation
from litebo.utils.limit import time_limit, TimeoutException
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
            if _result is None:
                result = MAXINT
            elif isinstance(_result, dict):
                result = _result['objective_value']
            else:
                result = _result
    except Exception as e:
        if isinstance(e, TimeoutException):
            trial_state = TIMEOUT
        else:
            traceback.print_exc(file=sys.stdout)
            trial_state = FAILED
        result = MAXINT
    return [config, result]


class pSMBO(BOBase):
    def __init__(self, objective_function, config_space,
                 batch_size=4,
                 sample_strategy='bo',
                 parallel_strategy='async',
                 time_limit_per_trial=180,
                 max_runs=200,
                 logging_dir='logs',
                 initial_configurations=None,
                 init_strategy='random_explore_first',
                 history_bo_data: List[OrderedDict] = None,
                 initial_runs=10,
                 task_id=None,
                 random_state=1):

        super().__init__(objective_function, config_space, task_id=task_id, output_dir=logging_dir,
                         random_state=random_state, initial_runs=initial_runs, max_runs=max_runs,
                         sample_strategy=sample_strategy, time_limit_per_trial=time_limit_per_trial,
                         history_bo_data=history_bo_data)
        if parallel_strategy == 'sync':
            self.config_advisor = SyncBatchAdvisor(config_space,
                                                   initial_trials=initial_runs,
                                                   initial_configurations=initial_configurations,
                                                   init_strategy=init_strategy,
                                                   optimization_strategy=sample_strategy,
                                                   batch_size=batch_size,
                                                   task_id=task_id,
                                                   output_dir=logging_dir,
                                                   rng=self.rng)
        elif parallel_strategy == 'async':
            self.config_advisor = AsyncBatchAdvisor(config_space,
                                                    initial_trials=initial_runs,
                                                    initial_configurations=initial_configurations,
                                                    init_strategy=init_strategy,
                                                    optimization_strategy=sample_strategy,
                                                    task_id=task_id,
                                                    output_dir=logging_dir,
                                                    rng=self.rng)
        else:
            raise ValueError('Invalid parallel strategy - %s.' % parallel_strategy)

        self.parallel_strategy = parallel_strategy
        self.batch_size = batch_size

    def callback(self, result):
        _config, _perf = result[0], result[1]
        _observation = [_config, _perf, SUCCESS]
        # Report the result, and remove the config from the running queue.
        self.config_advisor.update_observation(_observation)
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

    def sync_run(self):
        with ParallelEvaluation(wrapper, n_worker=self.batch_size) as proc:
            batch_num = (self.max_iterations + self.batch_size - 1) // self.batch_size
            if self.batch_size > self.config_advisor.init_num:
                batch_num += 1  # fix bug
            batch_id = 0
            while batch_id < batch_num:
                configs = self.config_advisor.get_suggestions()
                params = [(self.objective_function, config, self.time_limit_per_trial) for config in configs]
                # Wait all workers to complete their corresponding jobs.
                results = proc.parallel_execute(params)
                # Report their results.
                for idx, (_config, _result) in enumerate(zip(configs, results)):
                    if _result[-1] is None:
                        _perf = MAXINT
                    elif isinstance(_result[-1], dict):
                        _perf = _result[-1]['objective_value']
                    else:
                        _perf = _result[-1]
                    _observation = [_config, _perf, SUCCESS]
                    self.config_advisor.update_observation(_observation)
                    self.logger.info('In the %d-th batch [%d], result is: %.3f' % (batch_id, idx, _perf))
                batch_id += 1

    def run(self):
        if self.parallel_strategy == 'async':
            self.async_run()
        else:
            self.sync_run()
