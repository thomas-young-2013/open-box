import psutil
import sys
import time
import traceback
from litebo.utils.constants import MAXINT, SUCCESS, FAILDED, TIMEOUT
from litebo.core.computation.parallel_process import ParallelEvaluation
from litebo.utils.limit import time_limit, TimeoutException
from litebo.core.sync_batch_advisor import SyncBatchAdvisor
from litebo.core.async_batch_advisor import AsyncBatchAdvisor
from litebo.optimizer.base import BOBase


def wrapper(param):
    print('worker proc', psutil.Process().pid)
    objective_function, config = param
    time_limit_per_trial = 5
    try:
        args, kwargs = (config,), dict()
        timeout_status, _result = time_limit(objective_function, time_limit_per_trial,
                                             args=args, kwargs=kwargs)
        if timeout_status:
            raise TimeoutException('Timeout: time limit for this evaluation is %.1fs' % time_limit_per_trial)
        else:
            result = _result if _result is not None else MAXINT
    except Exception as e:
        if isinstance(e, TimeoutException):
            trial_state = TIMEOUT
        else:
            traceback.print_exc(file=sys.stdout)
            trial_state = FAILDED
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
                 initial_runs=3,
                 task_id=None,
                 random_state=1):

        super().__init__(objective_function, config_space, task_id=task_id, output_dir=logging_dir,
                         random_state=random_state, initial_runs=initial_runs, max_runs=max_runs,
                         sample_strategy=sample_strategy, time_limit_per_trial=time_limit_per_trial)
        if parallel_strategy == 'sync':
            self.config_advisor = SyncBatchAdvisor(config_space,
                                                   initial_trials=initial_runs,
                                                   initial_configurations=initial_configurations,
                                                   optimization_strategy=sample_strategy,
                                                   task_id=task_id,
                                                   output_dir=logging_dir,
                                                   rng=self.rng)
        elif parallel_strategy == 'async':
            self.config_advisor = AsyncBatchAdvisor(config_space,
                                                    initial_trials=initial_runs,
                                                    initial_configurations=initial_configurations,
                                                    optimization_strategy=sample_strategy,
                                                    task_id=task_id,
                                                    output_dir=logging_dir,
                                                    rng=self.rng)
        else:
            raise ValueError('Invalid parallel strategy - %s.' % parallel_strategy)

        self.parallel_strategy = parallel_strategy
        self.batch_size = batch_size

    def callback(self, result):
        print('in Callback', psutil.Process().pid)
        print('callback', result)
        _config, _perf = result[0], result[1]
        # print('Add observation [%s, %.f]' % (str(result[0]), result[1]))
        _observation = [_config, _perf, SUCCESS]
        self.config_advisor.update_observation(_observation)
        self.iteration_id += 1
        print('in proc-%d' % psutil.Process().pid, self.iteration_id)

    def async_run(self):
        # print('main proc', psutil.Process().pid)

        with ParallelEvaluation(wrapper, n_worker=self.batch_size) as proc:
            while self.iteration_id < self.max_iterations:
                cnt = 0
                while self.iteration_id < self.max_iterations:
                    _config = self.config_advisor.get_suggestion()
                    _param = [self.objective_function, _config]
                    proc.process_pool.apply_async(wrapper, (_param,), callback=self.callback)
                print('current iteration id', self.iteration_id)
                print('in Main-proc-%d' % psutil.Process().pid, self.iteration_id)

    def run(self):
        if self.parallel_strategy == 'async':
            self.async_run()
        else:
            with ParallelEvaluation(wrapper, n_worker=4) as proc:
                while self.iteration_id < self.max_iterations:
                    configs = self.config_advisor.get_suggestions()
                    params = [(self.objective_function, config) for config in configs]
                    results = proc.parallel_execute(params)
                    for idx, (_config, _result) in enumerate(zip(configs, results)):
                        _observation = [_config, _result, SUCCESS]
                        self.config_advisor.update_observation(_observation)
                        print('In iteration %d-%d' % (self.iteration_id, idx), _result)
                    self.iteration_id += 1
