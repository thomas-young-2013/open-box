import abc
import logging
import numpy as np
from litebo.utils.history_container import HistoryContainer
from litebo.acquisition_function.acquisition import EI
from litebo.model.rf_with_instances import RandomForestWithInstances
from litebo.optimizer.ei_optimization import InterleavedLocalAndRandomSearch, RandomSearch
from litebo.optimizer.random_configuration_chooser import ChooserProb
from litebo.utils.util_funcs import get_types, get_rng
from litebo.config_space.util import convert_configurations_to_array
from litebo.utils.constants import MAXINT, SUCCESS, FAILDED, TIMEOUT


class BaseFacade(object, metaclass=abc.ABCMeta):
    def __init__(self, config_space, task_id):
        self.logger = logging.getLogger(self.__module__ + "." + self.__class__.__name__)
        self.history_container = HistoryContainer(task_id)
        self.config_space = config_space

    @abc.abstractmethod
    def run(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def iterate(self):
        raise NotImplementedError()

    def get_history(self):
        return self.history_container

    def get_incumbent(self):
        return self.history_container.get_incumbents()


class BayesianOptimization(BaseFacade):
    def __init__(self, objective_function, config_space, max_runs=200, task_id=None, rng=None):
        super().__init__(config_space, task_id)
        if rng is None:
            run_id, rng = get_rng()

        self.init_num = 3
        self.max_iterations = max_runs
        self.iteration_id = 0
        self.sls_max_steps = None
        self.sls_n_steps_plateau_walk = 10

        self.configurations = list()
        self.perfs = list()

        # Initialize the basic component in BO.
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
                n_steps_plateau_walk=self.sls_n_steps_plateau_walk
            )
        self._random_search = RandomSearch(
            self.acquisition_function, self.config_space, rng
        )
        self.random_configuration_chooser = ChooserProb(prob=0.5, rng=rng)

    def run(self):
        while self.iteration_id < self.max_iterations:
            self.iterate()

    def iterate(self):
        if len(self.configurations) == 0:
            X = np.array([])
        else:
            X = convert_configurations_to_array(self.configurations)
        Y = np.array(self.perfs, dtype=np.float64)
        config = self.choose_next(X, Y)

        trial_state = SUCCESS
        trial_info = None
        # TODO: how to skip this.
        if config not in self.configurations:
            # Evaluate this configuration.
            try:
                perf = self.objective_function(config)
            except Exception as e:
                perf = MAXINT
                trial_info = str(e)
                trial_state = FAILDED
            self.configurations.append(config)
            self.perfs.append(perf)
            self.history_container.add(config, perf)
        else:
            self.logger.info('This configuration has been evaluated! Skip it.')
            config_idx = self.configurations.index(config)
            perf = self.perfs[config_idx]

        self.iteration_id += 1
        self.logger.info('Iteration %d, evaluation result: %.4f' % (self.iteration_id, perf))
        return trial_state, trial_info

    def choose_next(self, X: np.ndarray, Y: np.ndarray):
        _config_num = X.shape[0]
        if _config_num < self.init_num:
            if _config_num == 0:
                return self.config_space.get_default_configuration()
            else:
                return self._random_search.maximize(runhistory=self.history_container, num_points=1)[0]

        self.model.train(X, Y)

        incumbent_value = self.history_container.get_incumbents()[0][1]

        self.acquisition_function.update(model=self.model, eta=incumbent_value, num_data=len(self.history_container.data))

        challengers = self.optimizer.maximize(
            runhistory=self.history_container,
            num_points=1000,
            random_configuration_chooser=self.random_configuration_chooser
        )
        return list(challengers)[0]
