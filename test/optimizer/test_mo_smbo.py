import os
import sys
import numpy as np
import math
from copy import deepcopy
import argparse

import ConfigSpace.hyperparameters as CSH

sys.path.insert(0, os.getcwd())
from litebo.optimizer.generic_smbo import SMBO
from litebo.config_space import ConfigurationSpace

from pygmo import hypervolume

parser = argparse.ArgumentParser()
parser.add_argument('--n', type=int, default=100)

args = parser.parse_args()
max_runs = args.n

referencePoint = [1e5] * 2    # must greater than max value of objective


def Currin(x):
    return float(((1 - math.exp(-0.5 * (1 / x[1]))) * (
                (2300 * pow(x[0], 3) + 1900 * x[0] * x[0] + 2092 * x[0] + 60) / (
                    100 * pow(x[0], 3) + 500 * x[0] * x[0] + 4 * x[0] + 20))))


def branin(x1):
    x = deepcopy(x1)
    x[0] = 15 * x[0] - 5
    x[1] = 15 * x[1]
    return float(
        np.square(x[1] - (5.1 / (4 * np.square(math.pi))) * np.square(x[0]) + (5 / math.pi) * x[0] - 6) + 10 * (
                    1 - (1. / (8 * math.pi))) * np.cos(x[0]) + 10)


def multi_objective_func(config):
    xs = config.get_dictionary()
    x0 = (xs['x0'] + 100) / 200
    x1 = xs['x1'] / 10
    x = [x0, x1]    # x0, x1 in [0, 1]. test scale in MOSMBO
    y1 = Currin(x)
    y2 = branin(x)
    res = dict()
    res['config'] = config
    res['objs'] = (y1, y2)
    res['constraints'] = None
    return res


cs = ConfigurationSpace()
# x1 = CSH.UniformFloatHyperparameter("x0", -100, 100)
x1 = CSH.UniformIntegerHyperparameter("x0", -100, 100)  # test int in MOSMBO
x2 = CSH.UniformFloatHyperparameter("x1", 0, 10)
#x3 = CSH.Constant('c', 123)    # test constant in MOSMBO, no use todo
cs.add_hyperparameters([x1, x2])


# Evaluate MESMO
bo = SMBO(multi_objective_func, cs, num_objs=2, max_runs=max_runs,
          surrogate_type='gp_rbf', acq_type='mesmo',
          time_limit_per_trial=60, logging_dir='logs')
bo.config_advisor.optimizer.random_chooser.prob = 0     # no random
print('MESMO', '='*30)
# bo.run()
for i in range(max_runs):
    config, trial_state, objs, trial_info = bo.iterate()
    print(objs, config)
    hv = hypervolume(bo.get_history().get_all_perfs()).compute(referencePoint)
    hv2 = hypervolume(bo.get_history().get_pareto_front()).compute(referencePoint)
    print('hypervolume =', hv, hv2)

# Evaluate the random search.
bo_r = SMBO(multi_objective_func, cs, num_objs=2, max_runs=max_runs,
            time_limit_per_trial=60, sample_strategy='random', logging_dir='logs')
print('Random', '='*30)
# bo.run()
for i in range(max_runs):
    config, trial_state, objs, trial_info = bo_r.iterate()
    print(objs, config)
    hv = hypervolume(bo_r.get_history().get_all_perfs()).compute(referencePoint)
    hv2 = hypervolume(bo_r.get_history().get_pareto_front()).compute(referencePoint)
    print('hypervolume =', hv, hv2)

# plot pareto front
import matplotlib.pyplot as plt

pf = np.asarray(bo.get_history().get_pareto_front())
plt.scatter(pf[:, 0], pf[:, 1], label='mesmo')
pf_r = np.asarray(bo_r.get_history().get_pareto_front())
plt.scatter(pf_r[:, 0], pf_r[:, 1], label='random', marker='x')

print(pf.shape[0], pf_r.shape[0])

plt.title('Pareto Front')
plt.xlabel('Objective 1 - Currin')
plt.ylabel('Objective 2 - branin')
plt.legend()
plt.show()
