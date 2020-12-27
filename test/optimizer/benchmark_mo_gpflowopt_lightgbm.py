"""
example cmdline:

python test/optimizer/benchmark_mo_gpflowopt_lightgbm.py --datasets spambase --n 200 --rep 1 --start_id 0

"""
import os
import sys
import time
from functools import partial

import numpy as np
import argparse
import pickle as pkl

import gpflow
import gpflowopt

sys.path.insert(0, os.getcwd())
from mo_benchmark_function import timeit
from test_utils import check_datasets, load_data
from mo_benchmark_function import LightGBM
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, f1_score, accuracy_score

parser = argparse.ArgumentParser()
parser.add_argument('--n', type=int, default=200)
parser.add_argument('--rep', type=int, default=1)
parser.add_argument('--start_id', type=int, default=0)
parser.add_argument('--datasets', type=str)  # todo one dataset only

args = parser.parse_args()
max_runs = args.n
rep = args.rep
start_id = args.start_id
mth = 'gpflowopt-hvpoi'

seeds = [4774, 3711, 7238, 3203, 4254, 2137, 1188, 4356, 517, 5887,
         9082, 4702, 4801, 8242, 7391, 1893, 4400, 1192, 5553, 9039]

dataset_str = args.datasets
dataset_list = dataset_str.split(',')
data_dir = './test/optimizer/data/'
check_datasets(dataset_list, data_dir)

dataset = dataset_list[0]  # todo one dataset only
# set problem
from mo_benchmark_function import get_setup_lightgbm

setup = get_setup_lightgbm(dataset)
# multi_objective_func = setup['multi_objective_func']
cs = setup['cs']
# run_nsgaii = setup['run_nsgaii']
problem_str = setup['problem_str']
num_inputs = setup['num_inputs']
num_objs = setup['num_objs']
referencePoint = setup['referencePoint']
real_hv = setup['real_hv']
time_limit_per_trial = 2 * setup['time_limit']


def multi_objective_func(config, x, y):
    config = np.atleast_2d(config)
    assert config.shape == (1, num_inputs)
    config = config.flatten()
    param = config.tolist()
    param[0] = int(param[0]) * 50   # n_estimators *= 50
    param[2] = int(param[2])        # num_leaves
    param[3] = int(param[3])        # min_child_samples
    print('config =', param)
    """
    Caution:
        from functools import partial
        multi_objective_func = partial(multi_objective_func, x=x, y=y)
    """
    start_time = time.time()

    model = LightGBM(*param)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=1)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    time_taken = time.time() - start_time
    acc = -balanced_accuracy_score(y_test, y_pred)  # minimize

    y1 = acc
    y2 = time_taken

    res = [y1, y2]

    if any(res[i] > referencePoint[i] for i in range(len(referencePoint))):
        print('[ERROR]=== objective evaluate error! objs =', res, 'referencePoint =', referencePoint)
        res = [ref - 1e-5 for ref in referencePoint]
    print('objs =', res)

    return np.array(res).reshape(1, num_objs)


_x, _y = load_data(dataset, data_dir)
multi_objective_func = partial(multi_objective_func, x=_x, y=_y)


# Setup input domain    # Caution: param order!!!
# n_estimators *= 50
domain = gpflowopt.domain.ContinuousParameter("n_estimators", 2, 20) + \
         gpflowopt.domain.ContinuousParameter("learning_rate", 1e-3, 0.3) + \
         gpflowopt.domain.ContinuousParameter("num_leaves", 31, 2047) + \
         gpflowopt.domain.ContinuousParameter("min_child_samples", 5, 30) + \
         gpflowopt.domain.ContinuousParameter("subsample", 0.7, 1) + \
         gpflowopt.domain.ContinuousParameter("colsample_bytree", 0.7, 1)
# Initial evaluations
init_num = 10
assert max_runs > init_num
# X_init = gpflowopt.design.RandomDesign(init_num, domain).generate()
X_init = gpflowopt.design.LatinHyperCube(init_num, domain).generate()

Y_init = np.vstack([multi_objective_func(X_init[i, :]) for i in range(init_num)])

with timeit('%s all' % (mth,)):
    for run_i in range(start_id, start_id + rep):
        seed = seeds[run_i]
        np.random.seed(seed)
        with timeit('%s %d %d' % (mth, run_i, seed)):
            # One model for each objective
            objective_models = [
                gpflow.gpr.GPR(X_init.copy(), Y_init[:, [i]].copy(), gpflow.kernels.Matern52(2, ARD=True))
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
            result = optimizer.optimize(multi_objective_func, n_iter=max_runs - init_num)

            pf = optimizer.acquisition.pareto.front.value
            # pf, dom = gpflowopt.pareto.non_dominated_sort(hvpoi.data[1])
            # print(hvpoi.data[1])

            # Save result
            data = optimizer.acquisition.data  # data=(X, Y)
            hv_diffs = []
            for i in range(data[1].shape[0]):
                hv = gpflowopt.pareto.Pareto(data[1][:i + 1]).hypervolume(referencePoint)
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
