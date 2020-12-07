import numpy as np

from litebo.optimizer.generic_smbo import SMBO
from litebo.config_space import ConfigurationSpace, Configuration, \
    UniformFloatHyperparameter, UniformIntegerHyperparameter


def rosenbrock(config):
    X = np.array(list(config.get_dictionary().values()))
    res = dict()
    res['objs'] = ((1-X[0])**2 + 100*(X[1]-X[0]**2)**2, )
    res['constraints'] = ((X[0] - 1)**3 - X[1] + 1, X[0] + X[1] - 2)
    return res
rosenbrock_params = {
    'float': {
        'x1': (-1.5, 1.5, 0),
        'x2': (-0.5, 2.5, 0.5)
    }
}
rosenbrock_cs = ConfigurationSpace()
rosenbrock_cs.add_hyperparameters([UniformFloatHyperparameter(e, *rosenbrock_params['float'][e]) for e in rosenbrock_params['float']])


def rao(config):
    X = np.array(list(config.get_dictionary().values()))
    res = dict()
    res['objs'] = (-3*X[0]-4*X[1], )
    res['constraints'] = (3*X[0] - X[1] - 12,
                          3*X[0] + 11*X[1] - 66)
    return res
rao_cs = ConfigurationSpace()
rao_cs.add_hyperparameter(UniformFloatHyperparameter('x1', 0, 10, 1))
rao_cs.add_hyperparameter(UniformIntegerHyperparameter('x2', 0, 10, 1))


bo = SMBO(rao, rao_cs,
          num_constraints=2,
          max_runs=200)
bo.run()

