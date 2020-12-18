import os
import sys
import time
import numpy as np
import argparse
import pickle as pkl

import gpflow
import gpflowopt

from mo_benchmark_function import timeit

# set problem
from mo_benchmark_function import get_setup_bc, branin, Currin
setup = get_setup_bc()
# multi_objective_func = setup['multi_objective_func']
cs = setup['cs']
run_nsgaii = setup['run_nsgaii']
problem_str = setup['problem_str']
num_inputs = setup['num_inputs']
num_objs = setup['num_objs']
referencePoint = setup['referencePoint']
real_hv = setup['real_hv']

def multi_objective_func(x):
    x = np.atleast_2d(x)
    assert x.shape == (1, 2)
    x = x.flatten()
    y1 = branin(x)
    y2 = Currin(x)
    return np.array([[y1, y2]])

parser = argparse.ArgumentParser()
parser.add_argument('--n', type=int, default=110)
parser.add_argument('--rep', type=int, default=1)
parser.add_argument('--start_id', type=int, default=0)

args = parser.parse_args()
max_runs = args.n
rep = args.rep
start_id = args.start_id
mth = 'gpflowopt-hvpoi'

seeds = [4774, 3711, 7238, 3203, 4254, 2137, 1188, 4356,  517, 5887,
         9082, 4702, 4801, 8242, 7391, 1893, 4400, 1192, 5553, 9039]


# Setup input domain
domain = gpflowopt.domain.ContinuousParameter('x0', 0, 1) + \
         gpflowopt.domain.ContinuousParameter('x1', 0, 1)
# Initial evaluations
init_num = 10
assert max_runs > init_num
#X_init = gpflowopt.design.RandomDesign(init_num, domain).generate()
X_init2 = gpflowopt.design.LatinHyperCube(init_num, domain).generate()
X_init = np.array([    # generate from LatinHyperCube(10)
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

with timeit('%s all' % (mth,)):
    for run_i in range(start_id, start_id + rep):
        seed = seeds[run_i]
        np.random.seed(seed)
        with timeit('%s %d %d' % (mth, run_i, seed)):
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
            #pf, dom = gpflowopt.pareto.non_dominated_sort(hvpoi.data[1])
            #print(hvpoi.data[1])

            # Save result
            data = optimizer.acquisition.data   # data=(X, Y)
            hv_diffs = []
            for i in range(data[1].shape[0]):
                hv = gpflowopt.pareto.Pareto(data[1][:i+1]).hypervolume(referencePoint)
                hv_diff = real_hv - hv
                hv_diffs.append(hv_diff)
            print(seed, mth, 'pareto num:', pf.shape[0])
            print(seed, 'real hv =', real_hv)
            print(seed, 'hv_diffs:', hv_diffs)

            timestamp = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
            dir_path = 'logs/mo_benchmark_%s_%d/%s/' % (problem_str, max_runs, mth)
            file = 'benchmark_%s_%04d_%s.pkl' % (mth, seed, timestamp)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            with open(os.path.join(dir_path, file), 'wb') as f:
                save_item = (hv_diffs, pf, data)
                pkl.dump(save_item, f)
            print(dir_path, file, 'saved!')

if rep == 1:
    import matplotlib.pyplot as plt
    # Run NSGA-II to get 'real' pareto front
    cheap_pareto_front = run_nsgaii()

    # plot pareto front
    plt.scatter(pf[:, 0], pf[:, 1], label=mth)
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

    hv = optimizer.acquisition.pareto.hypervolume(np.array(referencePoint))
    hv_diff = real_hv - hv
    print(hv, hv_diff)
