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
parser.add_argument('--rand_prob', type=float, default=0)
parser.add_argument('--sample_num', type=int, default=1)

args = parser.parse_args()
max_runs = args.n
rand_prob = args.rand_prob
sample_num = args.sample_num

num_inputs = 2
num_objs = 2
referencePoint = [1.5] * 2    # must greater than max value of objective


def vlmop2(x):
    transl = 1 / np.sqrt(2)
    part1 = (x[0] - transl) ** 2 + (x[1] - transl) ** 2
    part2 = (x[0] + transl) ** 2 + (x[1] + transl) ** 2
    y1 = 1 - np.exp(-1 * part1)
    y2 = 1 - np.exp(-1 * part2)
    #return y1-1000, y2-1000
    return y1, y2


def multi_objective_func(config):
    xs = config.get_dictionary()
    x0 = xs['x0']
    x1 = xs['x1']
    x = [x0, x1]
    y1, y2 = vlmop2(x)
    res = dict()
    res['config'] = config
    res['objs'] = (y1, y2)
    res['constraints'] = None
    return res


search_range = 5

cs = ConfigurationSpace()
x0 = CSH.UniformFloatHyperparameter("x0", -search_range, search_range)
x1 = CSH.UniformFloatHyperparameter("x1", -search_range, search_range)
cs.add_hyperparameters([x0, x1])


# Evaluate MESMO
bo = SMBO(multi_objective_func, cs, num_objs=num_objs, max_runs=max_runs,
          surrogate_type='gp_rbf', acq_type='mesmo',
          time_limit_per_trial=60, logging_dir='logs')
bo.config_advisor.optimizer.random_chooser.prob = rand_prob     # set rand_prob, default 0
bo.config_advisor.acquisition_function.sample_num = sample_num  # set sample_num
print('MESMO', '='*30)
# bo.run()
for i in range(max_runs):
    config, trial_state, objs, trial_info = bo.iterate()
    print(objs, config)
    hv = hypervolume(bo.get_history().get_all_perfs()).compute(referencePoint)
    hv2 = hypervolume(bo.get_history().get_pareto_front()).compute(referencePoint)
    print('hypervolume =', hv, hv2)

# Evaluate the random search.
bo_r = SMBO(multi_objective_func, cs, num_objs=num_objs, max_runs=max_runs,
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
problem = Problem(num_inputs, num_objs)
problem.types[:] = Real(-search_range, search_range)
problem.function = vlmop2
algorithm = NSGAII(problem)
algorithm.run(2500)
cheap_pareto_front = np.array([list(solution.objectives) for solution in algorithm.result])


# plot pareto front
import matplotlib.pyplot as plt

pf = np.asarray(bo.get_history().get_pareto_front())
plt.scatter(pf[:, 0], pf[:, 1], label='mesmo')
pf_r = np.asarray(bo_r.get_history().get_pareto_front())
plt.scatter(pf_r[:, 0], pf_r[:, 1], label='random', marker='x')

plt.scatter(cheap_pareto_front[:, 0], cheap_pareto_front[:, 1], label='NSGA-II', marker='.', alpha=0.5)

print(pf.shape[0], pf_r.shape[0])

plt.title('Pareto Front')
plt.xlabel('Objective 1')
plt.ylabel('Objective 2')
plt.legend()
plt.show()
