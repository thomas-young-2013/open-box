import sys
import traceback
from typing import List
from collections import OrderedDict
from litebo.optimizer.base import BOBase
from litebo.utils.constants import MAXINT, SUCCESS, FAILED, TIMEOUT
from litebo.utils.limit import time_limit, TimeoutException


class SMBO(BOBase):
    def __init__(self, objective_function: callable, config_space,
                 sample_strategy: str = 'bo',
                 max_runs=200,
                 time_limit_per_trial=180,
                 advisor_type='default',
                 surrogate_type='prf',
                 history_bo_data: List[OrderedDict] = None,
                 logging_dir='logs',
                 initial_configurations=None,
                 initial_runs=3,
                 task_id=None,
                 random_state=1):

        super().__init__(objective_function, config_space, task_id=task_id, output_dir=logging_dir,
                         random_state=random_state, initial_runs=initial_runs, max_runs=max_runs,
                         sample_strategy=sample_strategy, time_limit_per_trial=time_limit_per_trial,
                         history_bo_data=history_bo_data)
        if advisor_type == 'default':
            from litebo.core.advisor import Advisor
            self.config_advisor = Advisor(config_space, initial_trials=initial_runs,
                                          initial_configurations=initial_configurations,
                                          optimization_strategy=sample_strategy,
                                          surrogate_type=surrogate_type,
                                          history_bo_data=history_bo_data,
                                          task_id=task_id,
                                          output_dir=logging_dir,
                                          rng=self.rng)
        elif advisor_type == 'tpe':
            from litebo.core.tpe_advisor import TPE_Advisor
            self.config_advisor = TPE_Advisor(config_space)

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
                    perf = _result if _result is not None else MAXINT
            except Exception as e:
                if isinstance(e, TimeoutException):
                    trial_state = TIMEOUT
                else:
                    traceback.print_exc(file=sys.stdout)
                    trial_state = FAILED
                perf = MAXINT
                trial_info = str(e)

            observation = [config, perf, trial_state]
            self.config_advisor.update_observation(observation)
        else:
            self.logger.info('This configuration has been evaluated! Skip it.')
            if config in self.config_advisor.configurations:
                config_idx = self.config_advisor.configurations.index(config)
                trial_state, perf = SUCCESS, self.config_advisor.perfs[config_idx]
            else:
                trial_state, perf = FAILED, MAXINT

        self.iteration_id += 1
        self.logger.info('In the %d-th iteration, the objective value: %.4f' % (self.iteration_id, perf))
        return config, trial_state, perf, trial_info
