import sys
import traceback
from typing import List
from collections import OrderedDict
from litebo.optimizer.base import BOBase
from litebo.utils.constants import MAXINT, SUCCESS, FAILED, TIMEOUT
from litebo.utils.limit import time_limit, TimeoutException
from litebo.core.base import Observation


"""
    The objective function returns a dictionary that has --- config, constraints, objs ---.
"""


class SMBO(BOBase):
    def __init__(self, objective_function: callable, config_space,
                 num_constraints=0,
                 num_objs=1,
                 sample_strategy: str = 'bo',
                 max_runs=200,
                 time_limit_per_trial=180,
                 advisor_type='default',
                 surrogate_type='prf',
                 acq_type='ei',
                 history_bo_data: List[OrderedDict] = None,
                 logging_dir='logs',
                 initial_configurations=None,
                 initial_runs=3,
                 task_id=None,
                 random_state=1):

        self.task_info = {'num_constraints': num_constraints, 'num_objs': num_objs}
        self.FAILED_PERF = [MAXINT] * num_objs
        super().__init__(objective_function, config_space, task_id=task_id, output_dir=logging_dir,
                         random_state=random_state, initial_runs=initial_runs, max_runs=max_runs,
                         sample_strategy=sample_strategy, time_limit_per_trial=time_limit_per_trial,
                         history_bo_data=history_bo_data)
        if advisor_type == 'default':
            from litebo.core.generic_advisor import Advisor
            self.config_advisor = Advisor(config_space, self.task_info,
                                          initial_trials=initial_runs,
                                          initial_configurations=initial_configurations,
                                          optimization_strategy=sample_strategy,
                                          surrogate_type=surrogate_type,
                                          acq_type=acq_type,
                                          history_bo_data=history_bo_data,
                                          task_id=task_id,
                                          output_dir=logging_dir,
                                          rng=self.rng)
        elif advisor_type == 'tpe':
            from litebo.core.tpe_advisor import TPE_Advisor
            self.config_advisor = TPE_Advisor(config_space)
        else:
            raise ValueError('Invalid advisor type!')

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
                    objs = _result['objs'] if _result['objs'] is not None else self.FAILED_PERF
                    constraints = _result['constraints']
            except Exception as e:
                if isinstance(e, TimeoutException):
                    trial_state = TIMEOUT
                else:
                    traceback.print_exc(file=sys.stdout)
                    trial_state = FAILED
                objs = self.FAILED_PERF
                constraints = None
                trial_info = str(e)

            observation = Observation(config, trial_state, constraints, objs)
            self.config_advisor.update_observation(observation)
        else:
            self.logger.info('This configuration has been evaluated! Skip it.')
            if config in self.config_advisor.configurations:
                config_idx = self.config_advisor.configurations.index(config)
                trial_state, objs = SUCCESS, self.config_advisor.perfs[config_idx]
            else:
                trial_state, objs = FAILED, self.FAILED_PERF

        self.iteration_id += 1
        self.logger.info('In the %d-th iteration, the objective value: %s' % (self.iteration_id, str(objs)))
        return config, trial_state, objs, trial_info
