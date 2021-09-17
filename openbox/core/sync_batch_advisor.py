# License: MIT

import copy
import numpy as np

from openbox.utils.config_space.util import convert_configurations_to_array
from openbox.utils.constants import MAXINT, SUCCESS
from openbox.core.generic_advisor import Advisor
from openbox.core.base import Observation


class SyncBatchAdvisor(Advisor):
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
                 task_id='default_task_id',
                 random_state=None):

        self.batch_size = batch_size
        self.batch_strategy = batch_strategy
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

    def get_suggestions(self, batch_size=None, history_container=None):
        if batch_size is None:
            batch_size = self.batch_size
        if history_container is None:
            history_container = self.history_container

        num_config_evaluated = len(history_container.configurations)
        num_config_successful = len(history_container.successful_perfs)

        if num_config_evaluated < self.init_num:
            if self.initial_configurations is not None:  # self.init_num equals to len(self.initial_configurations)
                next_configs = self.initial_configurations[num_config_evaluated: num_config_evaluated + batch_size]
                if len(next_configs) < batch_size:
                    next_configs.extend(
                        self.sample_random_configs(batch_size - len(next_configs), history_container))
                return next_configs
            else:
                return self.sample_random_configs(batch_size, history_container)

        if self.optimization_strategy == 'random':
            return self.sample_random_configs(batch_size, history_container)

        if num_config_successful < max(self.init_num, 1):
            self.logger.warning('No enough successful initial trials! Sample random configurations.')
            return self.sample_random_configs(batch_size, history_container)

        X = convert_configurations_to_array(history_container.configurations)
        Y = history_container.get_transformed_perfs()
        # cY = history_container.get_transformed_constraint_perfs()

        batch_configs_list = list()

        if self.batch_strategy == 'median_imputation':
            # set bilog_transform=False to get real cY for estimating median
            cY = history_container.get_transformed_constraint_perfs(bilog_transform=False)

            estimated_y = np.median(Y, axis=0).reshape(-1).tolist()
            estimated_c = np.median(cY, axis=0).tolist() if self.num_constraints > 0 else None
            batch_history_container = copy.deepcopy(history_container)

            for batch_i in range(batch_size):
                # use super class get_suggestion
                curr_batch_config = super().get_suggestion(batch_history_container)

                # imputation
                observation = Observation(curr_batch_config, SUCCESS, estimated_c, estimated_y, None)
                batch_history_container.update_observation(observation)
                batch_configs_list.append(curr_batch_config)

        elif self.batch_strategy == 'local_penalization':
            # local_penalization only supports single objective with no constraint
            self.surrogate_model.train(X, Y)
            incumbent_value = history_container.get_incumbents()[0][1]
            # L = self.estimate_L(X)
            for i in range(batch_size):
                self.acquisition_function.update(model=self.surrogate_model, eta=incumbent_value,
                                                 num_data=len(history_container.data),
                                                 batch_configs=batch_configs_list)

                challengers = self.optimizer.maximize(
                    runhistory=history_container,
                    num_points=5000,
                )
                batch_configs_list.append(challengers.challengers[0])
        else:
            raise ValueError('Invalid sampling strategy - %s.' % self.batch_strategy)
        return batch_configs_list
