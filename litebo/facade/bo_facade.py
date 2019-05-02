import abc
import logging
import numpy as np
from litebo.utils.history_container import HistoryContainer
from litebo.acquisition_function.acquisition import EI
from litebo.model.rf_with_instances import RandomForestWithInstances
from litebo.optimizer.ei_optimization import InterleavedLocalAndRandomSearch, RandomSearch
from litebo.optimizer.random_configuration_chooser import ChooserProb
from litebo.utils.util_funcs import get_types, get_rng
from litebo.configspace.util import convert_configurations_to_array
from litebo.utils.constants import *


class BaseFacade(object, metaclass=abc.ABCMeta):
    def __init__(self, configspace, task_id):
        self.logger = logging.getLogger(self.__module__ + "." + self.__class__.__name__)
        self.history_container = HistoryContainer(task_id)
        self.configspace = configspace

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
    def __init__(self, objective_function, configspace, max_runs=200, task_id=None, rng=None):
        super().__init__(configspace, task_id)
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
        types, bounds = get_types(configspace)
        # TODO: what is the feature array.
        self.model = RandomForestWithInstances(types=types, bounds=bounds, seed=rng.randint(MAXINT))
        self.acquisition_function = EI(self.model)
        self.optimizer = InterleavedLocalAndRandomSearch(
                acquisition_function=self.acquisition_function,
                config_space=self.configspace,
                rng=np.random.RandomState(seed=rng.randint(MAXINT)),
                max_steps=self.sls_max_steps,
                n_steps_plateau_walk=self.sls_n_steps_plateau_walk
            )
        self._random_search = RandomSearch(
            self.acquisition_function, self.configspace, rng
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

        # Evaluate this configuration.
        perf = self.objective_function(config)
        self.configurations.append(config)
        self.perfs.append(perf)
        self.history_container.add(config, perf)
        self.iteration_id += 1
        self.logger.info('In %d-th iteration, the perf is %.4f' % (self.iteration_id, perf))

    def choose_next(self, X: np.ndarray, Y: np.ndarray):
        if X.shape[0] < self.init_num:
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
