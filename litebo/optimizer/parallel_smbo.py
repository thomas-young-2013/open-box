import time
from litebo.utils.constants import MAXINT, SUCCESS, FAILDED, TIMEOUT
from litebo.core.computation.parallel_process import ParallelEvaluation
from litebo.utils.limit import time_limit, TimeoutException
from litebo.utils.logging_utils import setup_logger, get_logger
from litebo.core.advisor import Advisor


def wrapper(param):
    objective_function, config = param
    result = objective_function(config)
    return [config, result]


class pSMBO(object):
    def __init__(self, objective_function, config_space,
                 sample_strategy='bo',
                 time_limit_per_trial=180,
                 max_runs=200,
                 logging_dir='logs',
                 initial_configurations=None,
                 initial_runs=3,
                 task_id=None,
                 rng=None):

        self.config_advisor = Advisor(config_space, initial_trials=initial_runs,
                                      initial_configurations=initial_configurations,
                                      optimization_strategy=sample_strategy,
                                      task_id=task_id,
                                      output_dir=logging_dir,
                                      rng=rng)

        self.init_num = initial_runs
        self.max_iterations = max_runs
        self.iteration_id = 0
        self.sample_strategy = sample_strategy
        self.objective_function = objective_function
        self.time_limit_per_trial = time_limit_per_trial

    def callback(self, result):
        # print('callback', result)
        _config, _perf = result[0], result[1]
        # print('Add observation [%s, %.f]' % (str(result[0]), result[1]))
        _observation = [_config, _perf, SUCCESS]
        self.config_advisor.update_observation(_observation)
        time.sleep(0.5)

    def async_run(self):
        with ParallelEvaluation(wrapper, n_worker=4) as proc:
            while self.iteration_id < self.max_iterations:
                config = self.config_advisor.get_suggestion()
                configs = [(self.objective_function, config)]
                results = proc.parallel_execute(configs, callback=self.callback)
                for (_param, _result) in zip(configs, results):
                    _config, _perf = _result
                    _observation = [_config, _perf, SUCCESS]

                print('In iteration %d' % self.iteration_id, _result)
                # self.config_advisor.update_observation(_observation)
                self.iteration_id += 1

    def run(self):
        with ParallelEvaluation(self.objective_function, n_worker=4) as proc:
            while self.iteration_id < self.max_iterations:
                config = self.config_advisor.get_suggestion()
                configs = [config, config, config, config]
                results = proc.parallel_execute(configs)
                for (_config, _result) in zip(configs, results):
                    _observation = [_config, _result, SUCCESS]

                print('In iteration %d' % self.iteration_id, _result)
                # Avoid repeated configurations, add it once.
                self.config_advisor.update_observation(_observation)
                self.iteration_id += 1
