import sys
import traceback
from typing import List
from collections import OrderedDict
from tqdm import tqdm
from litebo.optimizer.base import BOBase
from litebo.utils.constants import MAXINT, SUCCESS, FAILED, TIMEOUT
from litebo.utils.limit import time_limit, TimeoutException
from litebo.utils.util_funcs import get_result
from litebo.utils.multi_objective import Hypervolume
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
                 **kwargs):

        if task_id is None:
            raise ValueError('Task id is not SPECIFIED. Please input task id first.')

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
                                          init_strategy=init_strategy,
                                          initial_configurations=initial_configurations,
                                          optimization_strategy=sample_strategy,
                                          surrogate_type=surrogate_type,
                                          acq_type=acq_type,
                                          acq_optimizer_type=acq_optimizer_type,
                                          ref_point=ref_point,
                                          history_bo_data=history_bo_data,
                                          task_id=task_id,
                                          output_dir=logging_dir,
                                          random_state=random_state)
        elif advisor_type == 'mcadvisor':
            from litebo.core.mc_advisor import MCAdvisor
            use_trust_region = kwargs.get('use_trust_region', False)
            self.config_advisor = MCAdvisor(config_space, self.task_info,
                                            mc_times=kwargs.get('mc_times', 10),
                                            initial_trials=initial_runs,
                                            init_strategy=init_strategy,
                                            initial_configurations=initial_configurations,
                                            optimization_strategy=sample_strategy,
                                            surrogate_type=surrogate_type,
                                            acq_type=acq_type,
                                            acq_optimizer_type=acq_optimizer_type,
                                            use_trust_region=use_trust_region,
                                            ref_point=ref_point,
                                            history_bo_data=history_bo_data,
                                            task_id=task_id,
                                            output_dir=logging_dir,
                                            random_state=random_state)
        elif advisor_type == 'tpe':
            from litebo.core.tpe_advisor import TPE_Advisor
            self.config_advisor = TPE_Advisor(config_space)
        else:
            raise ValueError('Invalid advisor type!')

    def run(self):
        for i in tqdm(range(self.iteration_id, self.max_iterations)):
            self.iterate()
        return self.get_history()

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
                    objs, constraints = get_result(_result, FAILED_PERF=self.FAILED_PERF)
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
        # Logging.
        self.logger.info('Iteration %d, objective value: %s' % (self.iteration_id, str(objs)))

        # Visualization.
        if isinstance(objs, (int, float)):
            objs = (objs, )
        for idx, obj in enumerate(objs):
            if obj < self.FAILED_PERF[idx]:
                self.writer.add_scalar('data/objective-%d' % (idx + 1), obj, self.iteration_id)
        return config, trial_state, objs, trial_info
