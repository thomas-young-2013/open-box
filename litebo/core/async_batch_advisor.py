import copy
import numpy as np

from litebo.config_space.util import convert_configurations_to_array
from litebo.utils.constants import MAXINT, SUCCESS, FAILDED, TIMEOUT
from litebo.core.advisor import Advisor


class AsyncBatchAdvisor(Advisor):
    def __init__(self, config_space,
                 batch_size=4,
                 initial_trials=3,
                 initial_configurations=None,
                 optimization_strategy='bo',
                 batch_strategy='median_imputation',
                 surrogate_type='prf',
                 output_dir='logs',
                 task_id=None,
                 rng=None):

        self.batch_size = batch_size
        self.batch_strategy = batch_strategy
        self.running_configs = list()
        super().__init__(config_space,
                         initial_trials=initial_trials,
                         initial_configurations=initial_configurations,
                         optimization_strategy=optimization_strategy,
                         surrogate_type=surrogate_type,
                         output_dir=output_dir,
                         task_id=task_id,
                         rng=rng)

        if batch_strategy == 'median_imputation':
            acq_type = 'ei'
        elif batch_strategy == 'local_penalization':
            acq_type = 'lpei'
        else:
            raise ValueError('Unsupported batch strategy - %s.' % batch_strategy)
        super(AsyncBatchAdvisor, self).setup_bo_basics(acq_type=acq_type)

    def create_initial_design(self, init_strategy='random'):
        default_config = self.config_space.get_default_configuration()
        if init_strategy == 'random':
            num_random_config = self.init_num * self.batch_size - 1
            initial_configs = [default_config] + self.sample_random_configs(num_random_config)
            return initial_configs
        else:
            raise ValueError('Unknown initial design strategy: %s.' % init_strategy)

    def get_suggestion(self):
        print('Current running configs', len(self.running_configs))

        if len(self.configurations) == 0:
            X = np.array([])
        else:
            failed_configs = list() if self.max_y is None else self.failed_configurations.copy()
            X = convert_configurations_to_array(self.configurations + failed_configs)

        # Failed trial.
        num_failed_trial = len(self.failed_configurations)
        failed_perfs = list() if self.max_y is None else [self.max_y] * num_failed_trial
        Y = np.array(self.perfs + failed_perfs, dtype=np.float64)
        all_considered_configs = self.configurations + self.failed_configurations + self.running_configs

        num_config_evaluated = len(all_considered_configs)
        if (num_config_evaluated < self.init_num) or self.optimization_strategy == 'random':
            _config = self.initial_configurations[num_config_evaluated]
            self.running_configs.append(_config)
            return _config

        if self.batch_strategy == 'median_imputation':
            X = convert_configurations_to_array(all_considered_configs)
            y_median = np.median(self.perfs)
            running_perfs = [y_median] * len(self.running_configs)
            Y = np.array(self.perfs + failed_perfs + running_perfs, dtype=np.float64)

            batch_history_container = copy.deepcopy(self.history_container)
            self.surrogate_model.train(X, Y)
            incumbent_value = batch_history_container.get_incumbents()[0][1]
            self.acquisition_function.update(model=self.surrogate_model, eta=incumbent_value,
                                             num_data=len(batch_history_container.data))

            challengers = self.optimizer.maximize(
                runhistory=batch_history_container,
                num_points=5000
            )

            is_repeated_config = True
            repeated_time = 0
            curr_batch_config = None
            while is_repeated_config:
                try:
                    curr_batch_config = challengers.challengers[repeated_time]
                    batch_history_container.add(curr_batch_config, y_median)
                except ValueError:
                    is_repeated_config = True
                    repeated_time += 1
                else:
                    is_repeated_config = False

            _config = curr_batch_config

        elif self.batch_strategy == 'local_penalization':
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

    def update_observation(self, observation):
        config, perf, trial_state = observation
        assert config in self.running_configs
        self.running_configs.remove(config)
        if not isinstance(perf, float):
            perf = perf[-1]
        if trial_state == SUCCESS and perf < MAXINT:
            if len(self.configurations) == 0:
                self.default_obj_value = perf

            self.configurations.append(config)
            self.perfs.append(perf)
            self.history_container.add(config, perf)

            self.perc = np.percentile(self.perfs, self.scale_perc)
            self.min_y = np.min(self.perfs)
            self.max_y = np.max(self.perfs)
        else:
            self.failed_configurations.append(config)
