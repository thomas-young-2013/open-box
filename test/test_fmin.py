import os
import sys
import numpy as np

# Import ConfigSpace and different types of parameters
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformFloatHyperparameter, UniformIntegerHyperparameter
from ConfigSpace.conditions import InCondition
sys.path.append(os.getcwd())
from litebo.facade.bo_facade import BayesianOptimization
from litebo.config_space import ConfigurationSpace


def branin(x):
    xs = x.get_dictionary()
    x1 = xs['x1']
    x2 = xs['x2']
    a = 1.
    b = 5.1 / (4.*np.pi**2)
    c = 5. / np.pi
    r = 6.
    s = 10.
    t = 1. / (8.*np.pi)
    ret = a*(x2-b*x1**2+c*x1-r)**2+s*(1-t)*np.cos(x1)+s
    import time
    time.sleep(5)
    return ret


cs = ConfigurationSpace()
x1 = UniformFloatHyperparameter("x1", -5, 10, default_value=0)
x2 = UniformFloatHyperparameter("x2", 0, 15, default_value=0)
cs.add_hyperparameters([x1, x2])

bo = BayesianOptimization(branin, cs, max_runs=50, time_limit_per_trial=3, logging_dir='logs')
bo.run()
inc_value = bo.get_incumbent()
print(bo.get_history().data)
print('='*30)
print(inc_value)
