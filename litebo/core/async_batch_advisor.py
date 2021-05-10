import copy
import numpy as np

from litebo.utils.config_space.util import convert_configurations_to_array
from litebo.utils.constants import MAXINT, SUCCESS
from litebo.core.base import Observation
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

    def get_suggestion(self, history_container=None):
        self.logger.info('#Call get_suggestion. len of running configs = %d.' % len(self.running_configs))

        if history_container is None:
            history_container = self.history_container

        num_config_all = len(history_container.configurations) + len(self.running_configs)
        num_config_successful = len(history_container.successful_perfs)

        if (num_config_all < self.init_num) or \
                num_config_successful < self.bo_start_n or \
                self.optimization_strategy == 'random':
            if num_config_all >= len(self.initial_configurations):
                _config = self.sample_random_configs(1, history_container)[0]
            else:
                _config = self.initial_configurations[num_config_all]
            self.running_configs.append(_config)
            return _config

        X = convert_configurations_to_array(history_container.configurations)
        Y = history_container.get_transformed_perfs()
        # cY = history_container.get_transformed_constraint_perfs()

        if self.batch_strategy == 'median_imputation':
            # set bilog_transform=False to get real cY for estimating median
            cY = history_container.get_transformed_constraint_perfs(bilog_transform=False)

            estimated_y = np.median(Y, axis=0).reshape(-1).tolist()
            estimated_c = np.median(cY, axis=0).tolist() if self.num_constraints > 0 else None
            batch_history_container = copy.deepcopy(history_container)
            # imputation
            for config in self.running_configs:
                observation = Observation(config, SUCCESS, estimated_c, estimated_y, None)
                batch_history_container.update_observation(observation)

            # use super class get_suggestion
            _config = super().get_suggestion(batch_history_container)

        elif self.batch_strategy == 'local_penalization':
            # local_penalization only supports single objective with no constraint
            self.surrogate_model.train(X, Y)
            incumbent_value = history_container.get_incumbents()[0][1]
            self.acquisition_function.update(model=self.surrogate_model, eta=incumbent_value,
                                             num_data=len(history_container.data),
                                             batch_configs=self.running_configs)

            challengers = self.optimizer.maximize(
                runhistory=history_container,
                num_points=5000
            )
            _config = challengers.challengers[0]
        else:
            raise ValueError('Invalid sampling strategy - %s.' % self.batch_strategy)

        self.running_configs.append(_config)
        return _config

    def update_observation(self, observation: Observation):
        config, trial_state, constraints, objs, elapsed_time = observation
        assert config in self.running_configs
        self.running_configs.remove(config)
        super().update_observation(observation)
