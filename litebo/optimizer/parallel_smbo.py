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
from litebo.core.ts_advisor import TS_Advisor
from litebo.optimizer.base import BOBase
from litebo.core.base import Observation


class pSMBO(BOBase):
    def __init__(self, objective_function, config_space,
                 batch_size=4,
                 num_constraints=0,
                 num_objs=1,
                 sample_strategy='bo',
                 parallel_strategy='async',
                 use_trust_region=False,
                 time_limit_per_trial=180,
                 max_runs=200,
                 logging_dir='logs',
                 initial_configurations=None,
                 init_strategy='random_explore_first',
                 history_bo_data: List[OrderedDict] = None,
                 initial_runs=10,
                 task_id=None,
                 random_state=1,
                 options=None):

        self.task_info = {'num_constraints': num_constraints, 'num_objs': num_objs}
        self.options = dict() if options is None else options
        self.FAILED_PERF = [MAXINT] * num_objs
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
        elif parallel_strategy == 'ts':
            self.use_trust_region = use_trust_region
            self.config_advisor = TS_Advisor(config_space,
                                             self.task_info,
                                             initial_trials=initial_runs,
                                             init_strategy=init_strategy,
                                             history_bo_data=history_bo_data,
                                             optimization_strategy=sample_strategy,
                                             surrogate_type='gp',
                                             output_dir=logging_dir,
                                             task_id=task_id,
                                             random_state=random_state,
                                             batch_size=batch_size,
                                             use_trust_region=self.use_trust_region)
        else:
            raise ValueError('Invalid parallel strategy - %s.' % parallel_strategy)

        self.parallel_strategy = parallel_strategy
        self.batch_size = batch_size

    def callback(self, result):
        config, trial_state, objs, constraints, trial_info = result
        # Report the result, and remove the config from the running queue.
        observation = Observation(config, trial_state, constraints, objs)
        self.config_advisor.update_observation(observation)
        # Parent process: collect the result and increment id.
        self.iteration_id += 1

    def wrapper(self, param):
        objective_function, config, time_limit_per_trial = param
        trial_state, trial_info = SUCCESS, None
        try:
            args, kwargs = (config,), dict()
            timeout_status, _result = time_limit(objective_function, time_limit_per_trial,
                                                 args=args, kwargs=kwargs)
            if timeout_status:
                raise TimeoutException('Timeout: time limit for this evaluation is %.1fs' % time_limit_per_trial)
            else:
                objs = _result['objs'] if _result['objs'] is not None else self.FAILED_PERF
                constraints = _result.get('constraints', None)
        except Exception as e:
            if isinstance(e, TimeoutException):
                trial_state = TIMEOUT
            else:
                traceback.print_exc(file=sys.stdout)
                trial_state = FAILED
            objs = self.FAILED_PERF
            constraints = None
            trial_info = str(e)
        return config, trial_state, objs, constraints, trial_info

    def async_run(self):
        with ParallelEvaluation(self.wrapper, n_worker=self.batch_size) as proc:
            while self.iteration_id < self.max_iterations:
                _config = self.config_advisor.get_suggestion()
                _param = [self.objective_function, _config, self.time_limit_per_trial]
                # Submit a job to worker.
                proc.process_pool.apply_async(self.wrapper, (_param,), callback=self.callback)
                while len(self.config_advisor.running_configs) >= self.batch_size:
                    # Wait for workers.
                    time.sleep(0.3)

    def sync_run(self):
        with ParallelEvaluation(self.wrapper, n_worker=self.batch_size) as proc:
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
                    self.logger.info('In the %d-th batch [%d], using config %s, result is: %.3f'
                                     % (batch_id, idx, str(_config), _perf))
                batch_id += 1

    def ts_run(self):
        with ParallelEvaluation(self.wrapper, n_worker=self.batch_size) as proc:
            batch_num = (self.max_iterations + self.batch_size - 1) // self.batch_size
            batch_id = 0
            while batch_id < batch_num:
                configs = self.config_advisor.get_suggestions()
                params = [(self.objective_function, config, self.time_limit_per_trial) for config in configs]

                # Wait for all workers to complete their corresponding jobs
                results = proc.parallel_execute(params)

                # Get incumbent value of the last batch
                incumbent_value = self.config_advisor.history_container.incumbent_value
                min_objs_batch = self.FAILED_PERF

                # Report the results
                if self.task_info['num_constraints'] == 0:
                    for idx, _result in enumerate(results):
                        config, trial_state, objs, constraints, trial_info = _result
                        observation = Observation(config, trial_state, constraints, objs)
                        self.config_advisor.update_observation(observation)
                        min_objs_batch = [min(obj, min_obj) for obj, min_obj in zip(objs, min_objs_batch)]
                        self.logger.info('In the %d-th batch [%d], result is: %s' % (batch_id, idx, str(objs)))
                    if self.use_trust_region:
                        # Adjust trust region
                        self.config_advisor.adjust_length(min_objs_batch, incumbent_value)
                        if self.config_advisor.length < self.config_advisor.length_min:
                            return
                else:
                    all_infeasible = True
                    closest_to_feasible = None
                    closest_violation = MAXINT
                    for idx, _result in enumerate(results):
                        config, trial_state, objs, constraints, trial_info = _result
                        observation = Observation(config, trial_state, constraints, objs)
                        self.config_advisor.update_observation(observation)
                        if all_infeasible:
                            violation = sum(c for c in constraints if c > 0)
                            if violation == 0:
                                all_infeasible = False
                                min_objs_batch = [min(obj, min_obj) for obj, min_obj in zip(objs, min_objs_batch)]
                            elif violation < closest_violation:
                                closest_to_feasible = config.get_array()
                                closest_violation = violation
                        self.logger.info('In the %d-th batch [%d], result is: %s' % (batch_id, idx, str(objs)))
                    if self.use_trust_region:
                        if all_infeasible:
                            # Set the closest to infeasible point as center if all infeasible
                            self.config_advisor.center = closest_to_feasible
                        else:
                            # Use the incumbent as center if not all infeasible
                            self.config_advisor.center = None
                        # Adjust trust region's length
                        self.config_advisor.adjust_length(min_objs_batch, incumbent_value)
                        if self.config_advisor.length < self.config_advisor.length_min:
                            print('Trust region method finished.')
                            return

                batch_id += 1

    def get_history(self):
        assert self.config_advisor is not None
        return self.config_advisor.history_container

    def get_incumbents(self):
        assert self.config_advisor is not None
        return self.config_advisor.history_container.get_incumbents()

    def run(self):
        if self.parallel_strategy == 'async':
            self.async_run()
        elif self.parallel_strategy == 'sync':
            self.sync_run()
        elif self.parallel_strategy == 'ts':
            self.ts_run()
