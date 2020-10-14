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
        trial_state_list = list()
        trial_info_list = list()
        perf_list = list()

        for i, config in enumerate(config_list):
            trial_state_list.append(SUCCESS)
            trial_info_list.append(None)
            perf_list.append(None)

            if config not in (self.configurations + self.failed_configurations):
                # Evaluate this configuration.
                try:
                    args, kwargs = (config,), dict()
                    timeout_status, _result = time_limit(self.objective_function, self.time_limit_per_trial,
                                                         args=args, kwargs=kwargs)
                    if timeout_status:
                        raise TimeoutException(
                            'Timeout: time limit for this evaluation is %.1fs' % self.time_limit_per_trial)
                    else:
                        perf_list[i] = _result
                except Exception as e:
                    if isinstance(e, TimeoutException):
                        trial_state_list[i] = TIMEOUT
                    else:
                        traceback.print_exc(file=sys.stdout)
                        trial_state_list[i] = FAILDED
                    perf_list[i] = MAXINT
                    trial_info_list[i] = str(e)
                    self.logger.error(trial_info_list[i])

                if trial_state_list[i] == SUCCESS and perf_list[i] < MAXINT:
                    if len(self.configurations) == 0:
                        self.default_obj_value = perf_list[i]

                    self.configurations.append(config)
                    self.perfs.append(perf_list[i])
                    self.history_container.add(config, perf_list[i])

                    self.perc = np.percentile(self.perfs, self.scale_perc)
                    self.min_y = np.min(self.perfs)
                    self.max_y = np.max(self.perfs)
                else:
                    self.failed_configurations.append(config)
            else:
                self.logger.debug('This configuration has been evaluated! Skip it.')
                if config in self.configurations:
                    config_idx = self.configurations.index(config)
                    trial_state_list[i], perf_list[i] = SUCCESS, self.perfs[config_idx]
                else:
                    trial_state_list[i], perf_list[i] = FAILDED, MAXINT

            self.iteration_id += 1
            self.logger.info(
                'Iteration-%d, objective improvement: %.4f' % (
                    self.iteration_id, max(0, self.default_obj_value - perf_list[i])))
        return config_list, trial_state_list, perf_list, trial_info_list

    def choose_next(self, X: np.ndarray, Y: np.ndarray):
        # Select a batch of configs to evaluate next.
        _config_num = X.shape[0]
        batch_configs_list = list()
        batch_configs_cnt = self.batch_size  # numbers of samples which have not been determined yet

        while _config_num < self.init_num and batch_configs_cnt:
            default_config = self.config_space.get_default_configuration()
            if default_config not in (self.configurations + self.failed_configurations):
                batch_configs_list.append(default_config)
            else:
                batch_configs_list.append(
                    self._random_search.maximize(runhistory=self.history_container, num_points=1)[0])
            batch_configs_cnt -= 1
            _config_num += 1

        if batch_configs_cnt == 0:
            return batch_configs_list

        while batch_configs_cnt:
            if self.sample_strategy == 'random':
                batch_configs_list.append(self.sample_config())
            elif self.sample_strategy == 'bo':
                if self.random_configuration_chooser.check(self.iteration_id):  # 以一定的概率回到随机采样上来
                    batch_configs_list.append(self.sample_config())
                else:
                    self.model.train(X, Y)

                    incumbent_value = self.history_container.get_incumbents()[0][1]

                    self.acquisition_function.update(model=self.model, eta=incumbent_value,
                                                     num_data=len(self.history_container.data))

                    challengers = self.optimizer.maximize(
                        runhistory=self.history_container,
                        num_points=5000,
                        random_configuration_chooser=self.random_configuration_chooser
                    )

                    batch_configs_list.append(challengers.challengers[0])
            else:
                raise ValueError('Invalid sampling strategy - %s.' % self.sample_strategy)

            batch_configs_cnt-=1

        return batch_configs_list

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
