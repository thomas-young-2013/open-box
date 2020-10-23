import sys
import traceback

from litebo.utils.constants import MAXINT, SUCCESS, FAILDED, TIMEOUT
from litebo.utils.limit import time_limit, TimeoutException
from litebo.utils.logging_utils import setup_logger, get_logger
from litebo.core.advisor import Advisor


class SMBO(object):
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

    def run(self):
        while self.iteration_id < self.max_iterations:
            self.iterate()

    def iterate(self):
        config = self.config_advisor.get_suggestion()

        trial_state, trial_info = SUCCESS, None

        if config not in (self.config_advisor.configurations + self.config_advisor.failed_configurations):
            try:
                args, kwargs = (config,), dict()
                timeout_status, _result = time_limit(self.objective_function,
                                                     self.time_limit_per_trial,
                                                     args=args, kwargs=kwargs)
                if timeout_status:
                    raise TimeoutException(
                        'Timeout: time limit for this evaluation is %.1fs' % self.time_limit_per_trial)
                else:
                    perf = _result
            except Exception as e:
                if isinstance(e, TimeoutException):
                    trial_state = TIMEOUT
                else:
                    traceback.print_exc(file=sys.stdout)
                    trial_state = FAILDED
                perf = MAXINT
                trial_info = str(e)

            observation = [config, perf, trial_state]
            self.config_advisor.update_observation(observation)
        else:
            print('This configuration has been evaluated! Skip it.')
            if config in self.config_advisor.configurations:
                config_idx = self.config_advisor.configurations.index(config)
                trial_state, perf = SUCCESS, self.config_advisor.perfs[config_idx]
            else:
                trial_state, perf = FAILDED, MAXINT

        self.iteration_id += 1
        print('In the %d-th iteration, the objective value: %.4f' % (self.iteration_id, perf))
        return config, trial_state, perf, trial_info
