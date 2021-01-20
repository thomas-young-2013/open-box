"""
example cmdline:

python test/optimizer/mo/benchmark_mo_gpflowopt_dtlz2.py --n 110 --x 12 --y 2 --rep 1 --start_id 0

"""
import os
import sys
import time
import numpy as np
import argparse
import pickle as pkl

import gpflow
import gpflowopt

sys.path.insert(0, os.getcwd())
from test.test_utils import timeit
from litebo.utils.multi_objective import Hypervolume

parser = argparse.ArgumentParser()
parser.add_argument('--n', type=int, default=110)
parser.add_argument('--x', type=int, default=12)
parser.add_argument('--y', type=int, default=2)
parser.add_argument('--rep', type=int, default=1)
parser.add_argument('--start_id', type=int, default=0)

args = parser.parse_args()
max_runs = args.n
num_inputs = args.x
num_objs = args.y
rep = args.rep
start_id = args.start_id
mth = 'gpflowopt-hvpoi'

seeds = [4774, 3711, 7238, 3203, 4254, 2137, 1188, 4356,  517, 5887,
         9082, 4702, 4801, 8242, 7391, 1893, 4400, 1192, 5553, 9039]

# set problem
from litebo.benchmark.objective_functions.synthetic import DTLZ2
problem = DTLZ2(num_inputs, num_objs, random_state=None)
problem_str = 'DTLZ2-%d-%d' % (num_inputs, num_objs)


def multi_objective_func(x):
    x = np.atleast_2d(x)
    assert x.shape == (1, num_inputs)
    x = x.flatten()
    result = problem(x, convert=False)
    ret = np.array([result['objs']])
    assert ret.shape == (1, num_objs)
    return ret


# Setup input domain
domain = gpflowopt.domain.ContinuousParameter('x1', 0, 1)
for i in range(1, num_inputs):
    domain += gpflowopt.domain.ContinuousParameter(f'x{i+1}', 0, 1)
# Initial evaluations
init_num = 10
assert max_runs > init_num
#X_init = gpflowopt.design.RandomDesign(init_num, domain).generate()
X_init = gpflowopt.design.LatinHyperCube(init_num, domain).generate()

Y_init = np.vstack([multi_objective_func(X_init[i, :]) for i in range(X_init.shape[0])])

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

            #pf = optimizer.acquisition.pareto.front.value
            #pf, dom = gpflowopt.pareto.non_dominated_sort(hvpoi.data[1])
            #print(hvpoi.data[1])
            pf = gpflowopt.pareto.Pareto(optimizer.acquisition.data[1]).front.value

            # Save result
            data = optimizer.acquisition.data   # data=(X, Y)
            hv_diffs = []
            for i in range(data[1].shape[0]):
                # hv = gpflowopt.pareto.Pareto(data[1][:i+1]).hypervolume(problem.ref_point)    # ref_point problem
                hv = Hypervolume(problem.ref_point).compute(data[1][:i+1])
                hv_diff = problem.max_hv - hv
                hv_diffs.append(hv_diff)
            print(seed, mth, 'pareto num:', pf.shape[0])
            print(seed, 'real hv =', problem.max_hv)
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

    plt.scatter(pf[:, 0], pf[:, 1], label=mth)
    plt.scatter(Y_init[:, 0], Y_init[:, 1], label='init', marker='x')
    plt.title('Pareto Front of DTLZ2-%d-%d' % (num_inputs, num_objs))
    plt.xlabel('Objective 1')
    plt.ylabel('Objective 2')
    plt.legend()
    plt.show()
