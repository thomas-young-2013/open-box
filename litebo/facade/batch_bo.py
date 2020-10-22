import sys
import traceback
import numpy as np
from litebo.acquisition_function.acquisition import EI
from litebo.model.rf_with_instances import RandomForestWithInstances
from litebo.optimizer.ei_optimization import InterleavedLocalAndRandomSearch, RandomSearch
from litebo.optimizer.random_configuration_chooser import ChooserProb
from litebo.utils.util_funcs import get_types, get_rng
from litebo.config_space.util import convert_configurations_to_array
from litebo.utils.constants import MAXINT, SUCCESS, FAILDED, TIMEOUT
from litebo.utils.limit import time_limit, TimeoutException
from litebo.facade.bo_facade import BaseFacade


class BatchBayesianOptimization(BaseFacade):
    def __init__(self, objective_function, config_space,
                 sample_strategy='bo',
                 time_limit_per_trial=180,
                 max_runs=200,
                 logging_dir='logs',
                 initial_configurations=None,
                 initial_runs=3,
                 batch_size=3,
                 task_id=None,
                 rng=None):
        super().__init__(config_space, task_id, output_dir=logging_dir)
        self.logger = super()._get_logger(self.__class__.__name__)
        if rng is None:
            run_id, rng = get_rng()

        self.batch_size = batch_size
        self.init_num = initial_runs
        self.max_iterations = max_runs
        self.iteration_id = 0
        self.sls_max_steps = None
        self.n_sls_iterations = 5
        self.sls_n_steps_plateau_walk = 10
        self.time_limit_per_trial = time_limit_per_trial
        self.default_obj_value = MAXINT
        self.sample_strategy = sample_strategy

        self.configurations = list()
        self.failed_configurations = list()
        self.perfs = list()

        # Initialize the basic component in BO.
        self.config_space.seed(rng.randint(MAXINT))
        self.objective_function = objective_function
        types, bounds = get_types(config_space)
        # TODO: what is the feature array.
        self.model = RandomForestWithInstances(types=types, bounds=bounds, seed=rng.randint(MAXINT))
        self.acquisition_function = EI(self.model)
        self.optimizer = InterleavedLocalAndRandomSearch(
            acquisition_function=self.acquisition_function,
            config_space=self.config_space,
            rng=np.random.RandomState(seed=rng.randint(MAXINT)),
            max_steps=self.sls_max_steps,
            n_steps_plateau_walk=self.sls_n_steps_plateau_walk,
            n_sls_iterations=self.n_sls_iterations
        )
        self._random_search = RandomSearch(
            self.acquisition_function, self.config_space, rng
        )
        self.random_configuration_chooser = ChooserProb(prob=0.25, rng=rng)

    def run(self):
        while self.iteration_id < self.max_iterations:
            self.iterate()

    def iterate(self):
        if len(self.configurations) == 0:
            X = np.array([])
        else:
            failed_configs = list() if self.max_y is None else self.failed_configurations.copy()
            X = convert_configurations_to_array(self.configurations + failed_configs)

        failed_perfs = list() if self.max_y is None else [self.max_y] * len(self.failed_configurations)
        Y = np.array(self.perfs + failed_perfs, dtype=np.float64)

        config_list = self.choose_next(X, Y)

        for config in config_list:
            trial_state = SUCCESS
            trial_info = None

            if config not in (self.configurations + self.failed_configurations):
                # Evaluate this configuration.
                try:
                    args, kwargs = (config,), dict()
                    timeout_status, _result = time_limit(self.objective_function, self.time_limit_per_trial,
                                                         args=args, kwargs=kwargs)
                    if timeout_status:
                        raise TimeoutException('Timeout: time limit for this evaluation is %.1fs' % self.time_limit_per_trial)
                    else:
                        perf = _result
                except Exception as e:
                    if isinstance(e, TimeoutException):
                        trial_state = TIMEOUT
                    else:
                        traceback.print_exc(file=sys.stdout)
                        trial_state = FAILDED
                    perf = MAXINT
                    trial_info = str(e)
                    self.logger.error(trial_info)

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
            else:
                self.logger.debug('This configuration has been evaluated! Skip it.')
                if config in self.configurations:
                    config_idx = self.configurations.index(config)
                    trial_state, perf = SUCCESS, self.perfs[config_idx]
                else:
                    trial_state, perf = FAILDED, MAXINT

        self.iteration_id += 1
        self.logger.info(
            'Iteration-%d, objective improvement: %.4f' % (self.iteration_id, max(0, self.default_obj_value - perf)))
        return config_list, trial_state, perf, trial_info

    def choose_next(self, X: np.ndarray, Y: np.ndarray):
        # Select a batch of configs to evaluate next.
        return list()

    def sample_config(self):
        config = None
        _sample_cnt, _sample_limit = 0, 10000
        while True:
            _sample_cnt += 1
            config = self.config_space.sample_configuration()
            if config not in (self.configurations + self.failed_configurations):
                break
            if _sample_cnt >= _sample_limit:
                config = self.config_space.sample_configuration()
                break
        return config
