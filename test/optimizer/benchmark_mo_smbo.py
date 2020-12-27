"""
example cmdline:

python test/optimizer/benchmark_mo_smbo.py --mth mesmo --sample_num 1 --n 110 --rep 1 --start_id 0

"""
import os
import sys
import time
import numpy as np
import argparse
import pickle as pkl
from pygmo import hypervolume

sys.path.insert(0, os.getcwd())
from litebo.optimizer.generic_smbo import SMBO
from litebo.config_space import Configuration
from mo_benchmark_function import timeit

# set problem
from mo_benchmark_function import get_setup_bc
setup = get_setup_bc()
multi_objective_func = setup['multi_objective_func']
cs = setup['cs']
run_nsgaii = setup['run_nsgaii']
problem_str = setup['problem_str']
num_inputs = setup['num_inputs']
num_objs = setup['num_objs']
referencePoint = setup['referencePoint']
real_hv = setup['real_hv']

parser = argparse.ArgumentParser()
parser.add_argument('--n', type=int, default=100)
parser.add_argument('--rand_prob', type=float, default=0)
parser.add_argument('--sample_num', type=int, default=1)
parser.add_argument('--opt_num_mc', type=int, default=1000)
parser.add_argument('--opt_num_opt', type=int, default=1000)
parser.add_argument('--rep', type=int, default=1)
parser.add_argument('--start_id', type=int, default=0)
parser.add_argument('--mth', type=str, default='mesmo')

args = parser.parse_args()
max_runs = args.n
rand_prob = args.rand_prob
sample_num = args.sample_num
opt_num_mc = args.opt_num_mc    # MESMO optimizer only
opt_num_opt = args.opt_num_opt  # MESMO optimizer only
rep = args.rep
start_id = args.start_id
mth = args.mth

seeds = [4774, 3711, 7238, 3203, 4254, 2137, 1188, 4356,  517, 5887,
         9082, 4702, 4801, 8242, 7391, 1893, 4400, 1192, 5553, 9039]

# Evaluate mth
X_init = np.array([
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
])  # use latin hypercube
X_init = [Configuration(cs, vector=X_init[i]) for i in range(X_init.shape[0])]

with timeit('%s all' % (mth,)):
    for run_i in range(start_id, start_id + rep):
        seed = seeds[run_i]
        with timeit('%s %d %d' % (mth, run_i, seed)):
            try:
                bo = SMBO(multi_objective_func, cs, num_objs=num_objs, max_runs=max_runs,
                          # surrogate_type='gp_rbf',    # use default
                          acq_type=mth,
                          initial_configurations=X_init, initial_runs=10,
                          time_limit_per_trial=60, logging_dir='logs', random_state=seed)
                bo.config_advisor.optimizer.random_chooser.prob = rand_prob     # set rand_prob, default 0
                bo.config_advisor.acquisition_function.sample_num = sample_num  # set sample_num
                bo.config_advisor.acquisition_function.random_state = seed      # set random_state
                bo.config_advisor.optimizer.num_mc = opt_num_mc     # MESMO optimizer only
                bo.config_advisor.optimizer.num_opt = opt_num_opt   # MESMO optimizer only
                print(seed, mth, '===== start =====')
                # bo.run()
                hv_diffs = []
                for i in range(max_runs):
                    config, trial_state, objs, trial_info = bo.iterate()
                    print(seed, i, objs, config)
                    hv = hypervolume(bo.get_history().get_pareto_front()).compute(referencePoint)
                    hv2 = hypervolume(bo.get_history().get_all_perfs()).compute(referencePoint)
                    print(seed, i, 'hypervolume =', hv, hv2)
                    hv_diff = real_hv - hv
                    hv_diffs.append(hv_diff)
                    print(seed, i, 'hv diff =', hv_diff)

                # Save result
                pf = np.asarray(bo.get_history().get_pareto_front())
                data = bo.get_history().data
                print(seed, mth, 'pareto num:', pf.shape[0])
                print(seed, 'real hv =', real_hv)
                print(seed, 'hv_diffs:', hv_diffs)

                timestamp = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
                dir_path = 'logs/mo_benchmark_%s_%d/%s-%d/' % (problem_str, max_runs, mth, sample_num)
                file = 'benchmark_%s-%d_%04d_%s.pkl' % (mth, sample_num, seed, timestamp)
                if not os.path.exists(dir_path):
                    os.makedirs(dir_path)
                with open(os.path.join(dir_path, file), 'wb') as f:
                    save_item = (hv_diffs, pf, data)
                    pkl.dump(save_item, f)
                print(dir_path, file, 'saved!')
            except Exception as e:
                print(mth, run_i, seed, 'run error:', e)

if rep == 1:
    import matplotlib.pyplot as plt
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
        hv_diff = real_hv - hv
        print('hv diff =', hv_diff)

    pf_r = np.asarray(bo_r.get_history().get_pareto_front())
    print('random pareto num:', pf_r.shape[0])

    # Run NSGA-II to get 'real' pareto front
    cheap_pareto_front = run_nsgaii()

    # Plot pareto front
    plt.scatter(pf[:, 0], pf[:, 1], label=mth)
    plt.scatter(pf_r[:, 0], pf_r[:, 1], label='random', marker='x')
    plt.scatter(cheap_pareto_front[:, 0], cheap_pareto_front[:, 1], label='NSGA-II', marker='.', alpha=0.5)

    plt.title('Pareto Front')
    plt.xlabel('Objective 1 - branin')
    plt.ylabel('Objective 2 - Currin')
    plt.legend()
    plt.show()
