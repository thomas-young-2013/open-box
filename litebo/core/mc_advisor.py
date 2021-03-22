import numpy as np

from litebo.core.base import build_acq_func, build_optimizer, build_surrogate, Observation
from litebo.core.generic_advisor import Advisor
from litebo.utils.config_space.util import convert_configurations_to_array
from litebo.utils.history_container import HistoryContainer, MOHistoryContainer
from litebo.utils.multi_objective import NondominatedPartitioning
from litebo.utils.trust_region import TurboState
from litebo.utils.util_funcs import get_types


class MCAdvisor(Advisor):
    def __init__(self, config_space, task_info,
                 mc_times=10,
                 initial_trials=10,
                 initial_configurations=None,
                 init_strategy='random_explore_first',
                 history_bo_data=None,
                 optimization_strategy='bo',
                 surrogate_type=None,
                 acq_type=None,
                 acq_optimizer_type='batchmc',
                 use_trust_region=False,
                 ref_point=None,
                 output_dir='logs',
                 task_id=None,
                 random_state=None):

        self.mc_times = mc_times
        self.use_trust_region = use_trust_region
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
                         use_trust_region=use_trust_region,
                         output_dir=output_dir,
                         task_id=task_id,
                         random_state=random_state)

    def check_setup(self):
        """
            check num_objs, num_constraints, acq_type, surrogate_type
        """
        assert isinstance(self.num_objs, int) and self.num_objs >= 1
        assert isinstance(self.num_constraints, int) and self.num_constraints >= 0

        if self.surrogate_type is None:
            self.surrogate_type = 'gp'
        assert self.surrogate_type in ['gp', ]  # MC sample method
        self.constraint_surrogate_type = 'gp'

        # Single objective
        if self.num_objs == 1:
            if self.num_constraints == 0:
                if self.acq_type is None:
                    self.acq_type = 'mcei'
                assert self.acq_type in ['mcei']
            else:
                if self.acq_type is None:
                    self.acq_type = 'mceic'
                assert self.acq_type in ['mceic'] or self.use_trust_region

        # Multi objective
        else:
            if self.num_constraints == 0:
                if self.acq_type is None:
                    self.acq_type = 'mcehvi'
                assert self.acq_type in ['mcparego', 'mcehvi']
            else:
                if self.acq_type is None:
                    self.acq_type = 'mcehvic'
                assert self.acq_type in ['mcparegoc', 'mcehvic'] or self.use_trust_region

            # Check reference point is provided for EHVI methods
            if 'ehvi' in self.acq_type and self.ref_point is None:
                raise ValueError('Must provide reference point to use EHVI method!')

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
                                                   constraint_models=self.constraint_models,
                                                   mc_times=self.mc_times, ref_point=self.ref_point)

        self.optimizer = build_optimizer(func_str=self.acq_optimizer_type,
                                         acq_func=self.acquisition_function,
                                         config_space=self.config_space,
                                         rng=self.rng)

        if self.use_trust_region:
            types, bounds = get_types(self.config_space)
            cont_dim = np.sum(types == 0)
            self.turbo_state = TurboState(cont_dim)
        else:
            self.turbo_state = None

    def get_suggestion(self):
        # Check if turbo needs to be restarted
        if self.use_trust_region and self.turbo_state.restart_triggered:
            self.configurations = list()
            self.failed_configurations = list()
            self.perfs = list()
            self.history_container.restart()
            print('-'*30)
            print('Restart!')

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
            cY = None
            if self.num_constraints > 0:
                cY = []
                for c in self.constraint_perfs:
                    failed_c = list() if num_failed_trial == 0 else [max(c)] * num_failed_trial
                    cY.append(np.array(c + failed_c, dtype=np.float64))

                for i, model in enumerate(self.constraint_models):
                    model.train(X, cY[i])

            # update acquisition function
            if self.num_objs == 1:  # MC-EI
                incumbent_value = self.history_container.get_incumbents()[0][1]
                self.acquisition_function.update(model=self.surrogate_model,
                                                 constraint_models=self.constraint_models,
                                                 eta=incumbent_value)
            else:  # MC-ParEGO or MC-EHVI
                if self.acq_type.startswith('mcparego'):
                    self.acquisition_function.update(model=self.surrogate_model,
                                                     constraint_models=self.constraint_models)
                elif self.acq_type.startswith('mcehvi'):
                    partitioning = NondominatedPartitioning(self.num_objs, Y)
                    cell_bounds = partitioning.get_hypercell_bounds(ref_point=self.ref_point)
                    self.acquisition_function.update(model=self.surrogate_model,
                                                     constraint_models=self.constraint_models,
                                                     cell_lower_bounds=cell_bounds[0],
                                                     cell_upper_bounds=cell_bounds[1])

            # optimize acquisition function
            challengers = self.optimizer.maximize(runhistory=self.history_container,
                                                  num_points=5000,
                                                  turbo_state=self.turbo_state)
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

    def update_observation(self, observation: Observation):
        super().update_observation(observation)
        if self.use_trust_region:
            config, trial_state, constraints, objs = observation
            if self.num_objs > 1:
                raise NotImplementedError()
            else:
                self.turbo_state.update(objs[0])
