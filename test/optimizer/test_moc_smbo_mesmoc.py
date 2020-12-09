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

from platypus import NSGAII, Problem, Real
from pygmo import hypervolume

parser = argparse.ArgumentParser()
parser.add_argument('--n', type=int, default=100)

args = parser.parse_args()
max_runs = args.n

num_inputs = 2
num_objs = 2
num_constraints = 2
referencePoint = [1e5] * num_objs    # must greater than max value of objective


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


# test scale
# scale1 = 100
# scale2 = 10
scale1 = 1
scale2 = 1


def multi_objective_func(config):
    xs = config.get_dictionary()
    x0 = (xs['x0'] + scale1) / (2 * scale1)
    x1 = xs['x1'] / scale2
    x = [x0, x1]    # x0, x1 in [0, 1]. test scale in MOSMBO
    y1 = Currin(x)
    y2 = branin(x)
    c = (7-y2, -y1)     # y1 >= 0, y2 >= 7
    res = dict()
    res['config'] = config
    res['objs'] = (y1, y2)
    MAXINT = 2 ** 31 - 1
    # if any(ci > 0 for ci in c):
    #     res['objs'] = (MAXINT, MAXINT)
    res['constraints'] = c
    return res


cs = ConfigurationSpace()
x1 = CSH.UniformFloatHyperparameter("x0", -scale1, scale1)
# x1 = CSH.UniformIntegerHyperparameter("x0", -scale1, scale1)  # test int in MOSMBO
x2 = CSH.UniformFloatHyperparameter("x1", 0, scale2)
#x3 = CSH.Constant('c', 123)    # test constant in MOSMBO, no use todo
cs.add_hyperparameters([x1, x2])


# Evaluate MESMOC
# bo = SMBO(multi_objective_func, cs, num_constraints=num_constraints, num_objs=num_objs, max_runs=max_runs,
#           surrogate_type='gp_rbf', acq_type='mesmoc',
#           time_limit_per_trial=60, logging_dir='logs')
bo = SMBO(multi_objective_func, cs, num_constraints=num_constraints, num_objs=num_objs, max_runs=max_runs,
          acq_type='mesmoc2',
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
bo_r = SMBO(multi_objective_func, cs, num_constraints=num_constraints, num_objs=num_objs, max_runs=max_runs,
            time_limit_per_trial=60, sample_strategy='random', logging_dir='logs')
print('Random', '='*30)
# bo.run()
for i in range(max_runs):
    config, trial_state, objs, trial_info = bo_r.iterate()
    print(objs, config)
    hv = hypervolume(bo_r.get_history().get_all_perfs()).compute(referencePoint)
    hv2 = hypervolume(bo_r.get_history().get_pareto_front()).compute(referencePoint)
    print('hypervolume =', hv, hv2)

# Run NSGA-II to get 'real' pareto front
def CMO(xi):
    xi = np.asarray(xi)
    y1 = Currin(xi)
    y2 = branin(xi)
    y = [y1, y2]
    y_c = [7-y2, -y1]
    return y, y_c
problem = Problem(num_inputs, num_objs, num_constraints)
problem.types[:] = Real(0, 1)
problem.constraints[:] = "<=0"
problem.function = CMO
algorithm = NSGAII(problem)
algorithm.run(2500)
cheap_pareto_front = np.array([list(solution.objectives) for solution in algorithm.result])
cheap_constraints_values = [list(solution.constraints) for solution in algorithm.result]

# plot pareto front
import matplotlib.pyplot as plt

pf = np.asarray(bo.get_history().get_pareto_front())
plt.scatter(pf[:, 0], pf[:, 1], label='mesmoc')
pf_r = np.asarray(bo_r.get_history().get_pareto_front())
plt.scatter(pf_r[:, 0], pf_r[:, 1], label='random', marker='x')

plt.scatter(cheap_pareto_front[:, 0], cheap_pareto_front[:, 1], label='NSGA-II', marker='.', alpha=0.5)

print(pf.shape[0], pf_r.shape[0])

plt.title('Pareto Front')
plt.xlabel('Objective 1 - Currin')
plt.ylabel('Objective 2 - branin')
plt.legend()
plt.show()
