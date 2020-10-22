import os
import abc
import numpy as np

from litebo.core.build_bo import build_acq_func, build_optimizer, build_surrogate
from litebo.utils.util_funcs import get_rng
from litebo.utils.history_container import HistoryContainer
from litebo.utils.logging_utils import setup_logger, get_logger
from litebo.config_space.util import convert_configurations_to_array
from litebo.utils.constants import MAXINT, SUCCESS, FAILDED, TIMEOUT


class Advisor(object, metaclass=abc.ABCMeta):
    def __init__(self, config_space,
                 initial_trials=3,
                 initial_configurations=None,
                 optimization_strategy='bo',
                 surrogate_type='prf',
                 output_dir='logs',
                 task_id=None,
                 rng=None):

        # Create output (logging) directory.
        # Init logging module.
        # Random seed generator.
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.logger = self._get_logger(self.__class__.__name__)
        if rng is None:
            run_id, rng = get_rng()
        self.rng = rng

        # Basic components in Advisor.
        self.optimization_strategy = optimization_strategy
        self.default_obj_value = MAXINT
        self.configurations = list()
        self.failed_configurations = list()
        self.perfs = list()
        self.scale_perc = 5
        self.perc = None
        self.min_y = None
        self.max_y = None

        # Init the basic ingredients in Bayesian optimization.
        self.surrogate_type = surrogate_type
        self.init_num = initial_trials
        self.config_space = config_space
        self.config_space.seed(rng.randint(MAXINT))

        if initial_configurations is not None and len(initial_configurations) > 0:
            self.initial_configurations = initial_configurations
            self.init_num = len(initial_configurations)
        else:
            self.initial_configurations = self.create_initial_design()
            self.init_num = len(self.initial_configurations)
        self.history_container = HistoryContainer(task_id)

        self.surrogate_model = None
        self.acquisition_function = None
        self.optimizer = None
        self.setup_bo_basics()

    def setup_bo_basics(self):
        self.surrogate_model = build_surrogate(func_str='prf', config_space=self.config_space, rng=self.rng)

        self.acquisition_function = build_acq_func(func_str='ei', model=self.surrogate_model)

        self.optimizer = build_optimizer(func_str='local_random',
                                         acq_func=self.acquisition_function,
                                         config_space=self.config_space,
                                         rng=self.rng)

    def _get_logger(self, name):
        logger_name = 'lite-bo-%s' % name
        setup_logger(os.path.join(self.output_dir, '%s.log' % str(logger_name)))
        return get_logger(logger_name)

    def create_initial_design(self, init_strategy='random'):
        default_config = self.config_space.get_default_configuration()
        if init_strategy == 'random':
            num_random_config = self.init_num - 1
            initial_configs = [default_config] + self.sample_random_configs(num_random_config)
            return initial_configs
        else:
            raise ValueError('Unknown initial design strategy: %s.' % init_strategy)

    def get_suggestion(self):
        if len(self.configurations) == 0:
            X = np.array([])
        else:
            failed_configs = list() if self.max_y is None else self.failed_configurations.copy()
            X = convert_configurations_to_array(self.configurations + failed_configs)

        num_failed_trial = len(self.failed_configurations)
        failed_perfs = list() if self.max_y is None else [self.max_y] * num_failed_trial
        Y = np.array(self.perfs + failed_perfs, dtype=np.float64)

        num_config_evaluated = len(self.perfs)
        if num_config_evaluated < self.init_num:
            return self.initial_configurations[num_config_evaluated]

        if self.optimization_strategy == 'random':
            return self.sample_random_configs(1)
        elif self.optimization_strategy == 'bo':
            self.surrogate_model.train(X, Y)
            incumbent_value = self.history_container.get_incumbents()[0][1]
            self.acquisition_function.update(model=self.surrogate_model,
                                             eta=incumbent_value,
                                             num_data=num_config_evaluated)
            challengers = self.optimizer.maximize(runhistory=self.history_container,
                                                  num_points=5000)
            return challengers.challengers[0]
        else:
            raise ValueError('Unknown optimization strategy: %s.' % self.optimization_strategy)

    def update_observation(self, observation):
        config, perf, trial_state = observation

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

    def sample_random_configs(self, num_configs=1):
        configs = list()
        sample_cnt = 0
        while len(configs) < num_configs:
            sample_cnt += 1
            config = self.config_space.sample_configuration()
            if config not in (self.configurations + self.failed_configurations + configs):
                configs.append(config)
                sample_cnt = 0
            else:
                sample_cnt += 1
            if sample_cnt >= 200:
                configs.append(config)
                sample_cnt = 0
        return configs
