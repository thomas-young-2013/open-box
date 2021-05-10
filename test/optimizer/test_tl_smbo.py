import os
import sys
import numpy as np

from ConfigSpace.hyperparameters import UniformFloatHyperparameter
sys.path.append(os.getcwd())
from openbox.optimizer.smbo import SMBO
from openbox.utils.config_space import ConfigurationSpace


def branin(x):
    xs = x.get_dictionary()
    x1 = xs['x1']
    x2 = xs['x2']
    a = 1.
    b = 5.1 / (4. * np.pi ** 2)
    c = 5. / np.pi
    r = 6.
    s = 10.
    t = 1. / (8. * np.pi)
    ret = a * (x2 - b * x1 ** 2 + c * x1 - r) ** 2 + s * (1 - t) * np.cos(x1) + s
    return ret


cs = ConfigurationSpace()
x1 = UniformFloatHyperparameter("x1", -5, 10, default_value=0)
x2 = UniformFloatHyperparameter("x2", 0, 15, default_value=0)
cs.add_hyperparameters([x1, x2])

bo = SMBO(branin, cs, surrogate_type='prf', max_runs=50, time_limit_per_trial=3, logging_dir='logs', random_state=1)
bo.run()
inc_value = bo.get_incumbent()
print('BO', '=' * 30)
print(inc_value)

hist1 = bo.get_history().data

bo = SMBO(branin, cs, surrogate_type='prf', max_runs=30, time_limit_per_trial=3, logging_dir='logs', random_state=2)
bo.run()
inc_value = bo.get_incumbent()
print('BO', '=' * 30)
print(inc_value)

hist2 = bo.get_history().data

tlbo = SMBO(branin, cs, history_bo_data=[hist1, hist2], surrogate_type='tlbo_rgpe_prf',
            max_runs=40, time_limit_per_trial=3, logging_dir='logs')
tlbo.run()
inc_value = tlbo.get_incumbent()
print('TLBO', '=' * 30)
print(inc_value)
