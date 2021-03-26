import numpy as np

from litebo.utils.config_space.util import convert_configurations_to_array
from litebo.utils.constants import MAXINT, SUCCESS
from litebo.core.base import Observation
from litebo.utils.multi_objective import get_chebyshev_scalarization, NondominatedPartitioning
from litebo.core.generic_advisor import Advisor


class AsyncBatchAdvisor(Advisor):
    def __init__(self, config_space,
                 task_info,
                 batch_size=4,
                 batch_strategy='median_imputation',
                 initial_trials=10,
                 initial_configurations=None,
                 init_strategy='random_explore_first',
                 history_bo_data=None,
                 optimization_strategy='bo',
                 surrogate_type=None,
                 acq_type=None,
                 acq_optimizer_type='local_random',
                 ref_point=None,
                 output_dir='logs',
                 task_id=None,
                 random_state=None):

        self.batch_size = batch_size
        self.batch_strategy = batch_strategy
        self.running_configs = list()
        self.bo_start_n = 3
        super().__init__(config_space,
                         task_info,
                         initial_trials=initial_trials,
                         initial_configurations=initial_configurations,
                         init_strategy=init_strategy,
                         history_bo_data=history_bo_data,
                         optimization_strategy=optimization_strategy,
                         surrogate_type=surrogate_type,
                         acq_type=acq_type,
                         acq_optimizer_type=acq_optimizer_type,
                         ref_point=ref_point,
                         output_dir=output_dir,
                         task_id=task_id,
                         random_state=random_state)

    def check_setup(self):
        super().check_setup()

        if self.batch_strategy is None:
            self.batch_strategy = 'median_imputation'

        assert self.batch_strategy in ['median_imputation', 'local_penalization']

        if self.num_objs > 1 or self.num_constraints > 0:
            # local_penalization only supports single objective with no constraint
            assert self.batch_strategy in ['median_imputation', ]

        if self.batch_strategy == 'local_penalization':
            self.acq_type = 'lpei'

    def get_suggestion(self):
        self.logger.info('#Call get_suggestion. len of running configs = %d.' % len(self.running_configs))
        if len(self.configurations) == 0:
            X = np.array([])
        else:
            failed_configs = list() if self.max_y is None else self.failed_configurations.copy()
            X = convert_configurations_to_array(self.configurations + failed_configs)

        # Failed trial.
        num_failed_trial = len(self.failed_configurations)
        failed_perfs = list() if self.max_y is None else [self.max_y] * num_failed_trial
        Y = np.array(self.perfs + failed_perfs, dtype=np.float64)
        cY = []
        if self.num_constraints > 0:
            for c in self.constraint_perfs:
                failed_c = list() if num_failed_trial == 0 else [max(c)] * num_failed_trial
                cY.append(np.array(c + failed_c, dtype=np.float64))
        all_considered_configs = self.configurations + self.failed_configurations + self.running_configs

        num_config_evaluated = len(all_considered_configs)
        if (num_config_evaluated < self.init_num) or \
                len(self.history_container.data) <= self.bo_start_n or \
                self.optimization_strategy == 'random':
            if num_config_evaluated >= len(self.initial_configurations):
                _config = self.sample_random_configs(1)[0]
            else:
                _config = self.initial_configurations[num_config_evaluated]
            self.running_configs.append(_config)
            return _config

        if self.batch_strategy == 'median_imputation':
            X = convert_configurations_to_array(all_considered_configs)
            estimated_y = np.median(self.perfs, axis=0)
            running_perfs = [estimated_y] * len(self.running_configs)
            Y = np.array(self.perfs + failed_perfs + running_perfs, dtype=np.float64)

            if self.num_constraints > 0:
                estimated_c = np.median(cY, axis=1)
                for i in range(len(cY)):
                    cY[i] = np.append(cY[i], [estimated_c[i]] * len(self.running_configs), axis=0)

            # train surrogate model
            if self.num_objs == 1:
                self.surrogate_model.train(X, Y)
            elif self.acq_type == 'parego':
                weights = self.rng.random_sample(self.num_objs)
                weights = weights / np.sum(weights)
                scalarized_obj = get_chebyshev_scalarization(weights, Y)
                self.surrogate_model.train(X, scalarized_obj(Y))
            else:  # multi-objectives
                for i in range(self.num_objs):
                    self.surrogate_model[i].train(X, Y[:, i])

            # train constraint model
            if self.num_constraints > 0:
                for i, model in enumerate(self.constraint_models):
                    model.train(X, cY[i])

            # update acquisition function
            if self.num_objs == 1:
                incumbent_value = self.history_container.get_incumbents()[0][1]
                self.acquisition_function.update(model=self.surrogate_model,
                                                 constraint_models=self.constraint_models,
                                                 eta=incumbent_value,
                                                 num_data=num_config_evaluated)
            else:  # multi-objectives
                mo_incumbent_value = self.history_container.get_mo_incumbent_value()
                if self.acq_type == 'parego':
                    self.acquisition_function.update(model=self.surrogate_model,
                                                     constraint_models=self.constraint_models,
                                                     eta=scalarized_obj(np.atleast_2d(mo_incumbent_value)),
                                                     num_data=num_config_evaluated)
                elif self.acq_type.startswith('ehvi'):
                    partitioning = NondominatedPartitioning(self.num_objs, Y)
                    cell_bounds = partitioning.get_hypercell_bounds(ref_point=self.ref_point)
                    self.acquisition_function.update(model=self.surrogate_model,
                                                     constraint_models=self.constraint_models,
                                                     cell_lower_bounds=cell_bounds[0],
                                                     cell_upper_bounds=cell_bounds[1])
                else:
                    self.acquisition_function.update(model=self.surrogate_model,
                                                     constraint_models=self.constraint_models,
                                                     constraint_perfs=cY,  # for MESMOC
                                                     eta=mo_incumbent_value,
                                                     num_data=num_config_evaluated,
                                                     X=X, Y=Y)

            # optimize acquisition function
            challengers = self.optimizer.maximize(
                runhistory=self.history_container,
                num_points=5000,
            )

            is_repeated_config = True
            repeated_time = 0
            curr_batch_config = None
            while is_repeated_config:
                curr_batch_config = challengers.challengers[repeated_time]
                if curr_batch_config in all_considered_configs:
                    is_repeated_config = True
                    repeated_time += 1
                else:
                    is_repeated_config = False
            _config = curr_batch_config

        elif self.batch_strategy == 'local_penalization':
            # local_penalization only supports single objective with no constraint
            self.surrogate_model.train(X, Y)
            incumbent_value = self.history_container.get_incumbents()[0][1]
            self.acquisition_function.update(model=self.surrogate_model, eta=incumbent_value,
                                             num_data=len(self.history_container.data),
                                             batch_configs=self.running_configs)

            challengers = self.optimizer.maximize(
                runhistory=self.history_container,
                num_points=5000
            )
            _config = challengers.challengers[0]
        else:
            raise ValueError('Invalid sampling strategy - %s.' % self.batch_strategy)
        self.running_configs.append(_config)
        return _config

    def update_observation(self, observation: Observation):
        config, trial_state, constraints, objs = observation
        assert config in self.running_configs
        self.running_configs.remove(config)
        super().update_observation(observation)
