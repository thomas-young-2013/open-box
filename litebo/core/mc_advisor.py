import os
import abc
import numpy as np
from typing import Iterable

from litebo.utils.util_funcs import get_types
from litebo.utils.logging_utils import get_logger
from litebo.utils.history_container import HistoryContainer, MOHistoryContainer
from litebo.utils.constants import MAXINT, SUCCESS
from litebo.utils.samplers import SobolSampler, LatinHypercubeSampler
from litebo.utils.multi_objective import get_chebyshev_scalarization
from litebo.utils.config_space.util import convert_configurations_to_array
from litebo.core.base import build_acq_func, build_optimizer, build_surrogate
from litebo.core.base import Observation
from litebo.core.generic_advisor import Advisor


class MCAdvisor(Advisor):
    def __init__(self, config_space, task_info,
                 initial_trials=10,
                 initial_configurations=None,
                 init_strategy='random_explore_first',
                 history_bo_data=None,
                 optimization_strategy='bo',
                 surrogate_type=None,
                 acq_type=None,
                 acq_optimizer_type='batchmc',
                 ref_point=None,
                 output_dir='logs',
                 task_id=None,
                 random_state=None):

        super().__init__(config_space, task_info,
                         initial_trials=initial_trials,
                         initial_configurations=initial_configurations,
                         init_strategy=init_strategy,
                         history_bo_data=history_bo_data,
                         optimization_strategy=optimization_strategy,
                         surrogate_type=surrogate_type,
                         acq_type=acq_type,
                         acq_optimizer_type='batchmc',
                         ref_point=ref_point,
                         output_dir=output_dir,
                         task_id=task_id,
                         random_state=random_state)

    def check_setup(self):
        """
            check num_objs, num_constraints, acq_type, surrogate_type
        """
        assert isinstance(self.num_objs, int) and self.num_objs >= 1
        assert isinstance(self.num_constraints, int) and self.num_constraints >= 0

        # single objective no constraint
        if self.num_objs == 1 and self.num_constraints == 0:
            if self.acq_type is None:
                self.acq_type = 'mcei'
            assert self.acq_type in ['mcei']
            if self.surrogate_type is None:
                self.surrogate_type = 'prf'

        # multi-objective with constraints
        elif self.num_objs > 1 and self.num_constraints > 0:
            if self.acq_type is None:
                self.acq_type = 'mcparego'
            assert self.acq_type in ['mcparego']
            if self.surrogate_type is None:
                self.surrogate_type = 'gp'
            if self.constraint_surrogate_type is None:
                self.constraint_surrogate_type = 'gp'

        # multi-objective no constraint
        elif self.num_objs > 1:
            if self.acq_type is None:
                self.acq_type = 'mcparego'
            assert self.acq_type in ['mcparego']
            if self.surrogate_type is None:
                self.surrogate_type = 'gp'

        # single objective with constraints
        elif self.num_constraints > 0:
            if self.acq_type is None:
                self.acq_type = 'mceic'
            assert self.acq_type in ['mceic']
            if self.surrogate_type is None:
                if self.acq_type == 'mceic':
                    self.surrogate_type = 'gp'
                else:
                    self.surrogate_type = 'prf'
            if self.constraint_surrogate_type is None:
                self.constraint_surrogate_type = 'gp'

    def setup_bo_basics(self):
        if self.num_objs == 1:
            self.surrogate_model = build_surrogate(func_str=self.surrogate_type,
                                                   config_space=self.config_space,
                                                   rng=self.rng,
                                                   history_hpo_data=self.history_bo_data)
        else:  # multi-objectives
            self.surrogate_model = [build_surrogate(func_str=self.surrogate_type,
                                                    config_space=self.config_space,
                                                    rng=self.rng,
                                                    history_hpo_data=self.history_bo_data)
                                    for _ in range(self.num_objs)]

        if self.num_constraints > 0:
            self.constraint_models = [build_surrogate(func_str=self.constraint_surrogate_type,
                                                      config_space=self.config_space,
                                                      rng=self.rng) for _ in range(self.num_constraints)]

        self.acquisition_function = build_acq_func(func_str=self.acq_type, model=self.surrogate_model,
                                                   constraint_models=self.constraint_models)

        self.optimizer = build_optimizer(func_str=self.acq_optimizer_type,
                                         acq_func=self.acquisition_function,
                                         config_space=self.config_space,
                                         rng=self.rng)

    def get_suggestion(self):
        if len(self.configurations) == 0:
            X = np.array([])
        else:
            failed_configs = list() if self.max_y is None else self.failed_configurations.copy()
            X = convert_configurations_to_array(self.configurations + failed_configs)

        num_failed_trial = len(self.failed_configurations)
        failed_perfs = list() if self.max_y is None else [self.max_y] * num_failed_trial
        Y = np.array(self.perfs + failed_perfs, dtype=np.float64)

        num_config_evaluated = len(self.perfs + self.failed_configurations)
        if num_config_evaluated < self.init_num:
            return self.initial_configurations[num_config_evaluated]

        if self.optimization_strategy == 'random':
            return self.sample_random_configs(1)[0]
        elif self.optimization_strategy == 'bo':
            # train surrogate model
            if self.num_objs == 1:
                self.surrogate_model.train(X, Y)
            else:  # multi-objectives
                for i in range(self.num_objs):
                    self.surrogate_model[i].train(X, Y[:, i])

            # train constraint model
            cX = None
            if self.num_constraints > 0:
                cX = []
                for c in self.constraint_perfs:
                    failed_c = list() if num_failed_trial == 0 else [max(c)] * num_failed_trial
                    cX.append(np.array(c + failed_c, dtype=np.float64))

                for i, model in enumerate(self.constraint_models):
                    model.train(X, cX[i])

            # update acquisition function
            if self.num_objs == 1:
                incumbent_value = self.history_container.get_incumbents()[0][1]
                self.acquisition_function.update(model=self.surrogate_model,
                                                 constraint_models=self.constraint_models,
                                                 eta=incumbent_value,
                                                 num_data=num_config_evaluated)
            else:  # multi-objectives
                mo_incumbent_value = self.history_container.get_mo_incumbent_value()
                self.acquisition_function.update(model=self.surrogate_model,
                                                 constraint_models=self.constraint_models,
                                                 constraint_perfs=cX,  # for MESMOC
                                                 eta=mo_incumbent_value,
                                                 num_data=num_config_evaluated,
                                                 X=X, Y=Y)

            # optimize acquisition function
            # TODO: Sobol sampler is so slow，here we use only a batch of points
            challengers = self.optimizer.maximize(runhistory=self.history_container,
                                                  num_points=100)
            is_repeated_config = True
            repeated_time = 0
            cur_config = None
            while is_repeated_config:
                cur_config = challengers.challengers[repeated_time]
                if cur_config in (self.configurations + self.failed_configurations):
                    repeated_time += 1
                else:
                    is_repeated_config = False
            return cur_config
        else:
            raise ValueError('Unknown optimization strategy: %s.' % self.optimization_strategy)
