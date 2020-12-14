import os
import sys
import numpy as np
import math
from copy import deepcopy
import argparse

import gpflow
import gpflowopt

from platypus import NSGAII, Problem, Real

parser = argparse.ArgumentParser()
parser.add_argument('--n', type=int, default=100)

args = parser.parse_args()
max_runs = args.n

num_inputs = 2
num_objs = 2
referencePoint = [1e5] * num_objs    # must greater than max value of objective


def branin(x1):
    x = deepcopy(x1)
    x = np.asarray(x)
    assert x.shape == (2, )
    x[0] = 15 * x[0] - 5
    x[1] = 15 * x[1]
    return float(
        np.square(x[1] - (5.1 / (4 * np.square(math.pi))) * np.square(x[0]) + (5 / math.pi) * x[0] - 6) + 10 * (
                    1 - (1. / (8 * math.pi))) * np.cos(x[0]) + 10)


def Currin(x):
    x = np.asarray(x)
    assert x.shape == (2, )
    try:
        return float(((1 - math.exp(-0.5 * (1 / x[1]))) * (
                    (2300 * pow(x[0], 3) + 1900 * x[0] * x[0] + 2092 * x[0] + 60) / (
                        100 * pow(x[0], 3) + 500 * x[0] * x[0] + 4 * x[0] + 20))))
    except Exception as e:
        return 300


def multi_objective_func(x):
    x = np.atleast_2d(x)
    assert x.shape == (1, 2)
    x = x.flatten()
    y1 = branin(x)
    y2 = Currin(x)
    return np.array([[y1, y2]])

# Setup input domain
domain = gpflowopt.domain.ContinuousParameter('x0', 0, 1) + \
         gpflowopt.domain.ContinuousParameter('x1', 0, 1)
# Initial evaluations
init_num = 10
assert max_runs > init_num
#X_init = gpflowopt.design.RandomDesign(init_num, domain).generate()
X_init = gpflowopt.design.LatinHyperCube(init_num, domain).generate()
X_init2 = np.array([    # generate from LatinHyperCube(10)
    [ 6.66666667e-01,  3.33333333e-01],
    [ 3.33333333e-01,  6.66666667e-01],
    [ 2.22222222e-01,  2.22222222e-01],
    [ 7.77777778e-01,  7.77777778e-01],
    [ 5.55555556e-01,  0             ],
    [ 0,               5.55555556e-01],
    [ 1.00000000e+00,  4.44444444e-01],
    [ 4.44444444e-01,  1.00000000e+00],
    [ 8.88888889e-01,  1.11111111e-01],
    [ 1.11111111e-01,  8.88888889e-01],
])

Y_init = np.vstack([multi_objective_func(X_init[i, :]) for i in range(init_num)])
# One model for each objective
objective_models = [gpflow.gpr.GPR(X_init.copy(), Y_init[:,[i]].copy(), gpflow.kernels.Matern52(2, ARD=True))
                    for i in range(Y_init.shape[1])]
for model in objective_models:
    model.likelihood.variance = 0.01

hvpoi = gpflowopt.acquisition.HVProbabilityOfImprovement(objective_models)
# First setup the optimization strategy for the acquisition function
# Combining MC step followed by L-BFGS-B
acquisition_opt = gpflowopt.optim.StagedOptimizer([gpflowopt.optim.MCOptimizer(domain, 1000),
                                                   gpflowopt.optim.SciPyOptimizer(domain)])

# Then run the BayesianOptimizer for (max_runs-init_num) iterations
optimizer = gpflowopt.BayesianOptimizer(domain, hvpoi, optimizer=acquisition_opt, verbose=True)
result = optimizer.optimize(multi_objective_func, n_iter=max_runs-init_num)

pf = optimizer.acquisition.pareto.front.value

# Run NSGA-II to get 'real' pareto front
def CMO(xi):
    xi = np.asarray(xi)
    y = [branin(xi), Currin(xi)]
    return y
problem = Problem(num_inputs, num_objs)
problem.types[:] = Real(0, 1)
problem.function = CMO
algorithm = NSGAII(problem)
algorithm.run(2500)
cheap_pareto_front = np.array([list(solution.objectives) for solution in algorithm.result])

# plot pareto front
import matplotlib.pyplot as plt
plt.scatter(pf[:, 0], pf[:, 1], label='gpflow-opt-hvpoi')
plt.scatter(Y_init[:, 0], Y_init[:, 1], label='init', marker='x')

plt.scatter(cheap_pareto_front[:, 0], cheap_pareto_front[:, 1], label='NSGA-II', marker='.', alpha=0.5)

print(pf.shape[0])

plt.title('Pareto Front')
plt.xlabel('Objective 1 - branin')
plt.ylabel('Objective 2 - Currin')
plt.legend()
plt.show()

print('X_init:', X_init)
print('Y_init:', Y_init)
