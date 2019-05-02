import abc
import logging
import numpy as np
from litebo.utils.history_container import HistoryContainer
from litebo.acquisition_function.acquisition import EI
from litebo.model.rf_with_instances import RandomForestWithInstances
from litebo.optimizer.ei_optimization import InterleavedLocalAndRandomSearch
from litebo.utils.util_funcs import get_types, get_rng
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
    def __init__(self, configspace, max_runs=200, task_id=None, rng=None):
        super().__init__(configspace, task_id)
        if rng is None:
            run_id, rng = get_rng()

        self.max_iterations = max_runs
        self.iteration_id = 0
        self.sls_max_steps = None
        self.sls_n_steps_plateau_walk = 10

        # Initialize the basic component in BO.
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

    def iterate(self):
        if self.iteration_id == 0:
            # Initialize the configurations.
            pass
        else:
            # Choose a configuration.
            pass
