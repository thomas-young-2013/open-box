"""
re-calculate pareto hyper-volume
"""

import os
import sys
import argparse
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt

sys.path.insert(0, os.getcwd())
from litebo.utils.history_container import MOHistoryContainer
from litebo.utils.multi_objective import Hypervolume

default_mths = 'mesmo-1,usemo-1,gpflowopt-hvpoi'
parser = argparse.ArgumentParser()
parser.add_argument('--n', type=int, default=100)
parser.add_argument('--mths', type=str, default=default_mths)
parser.add_argument('--recalc', type=int, default=1)

args = parser.parse_args()
max_runs = args.n
mths = args.mths.split(',')     # list of 'mth-sample_num'
recalc = args.recalc    # recalculate hypervolume
# referencePoint = [1e5] * 2
# real_hv = 1e10
# referencePoint = [20, 12]
# real_hv = 20*12
referencePoint = [301, 200]
real_hv = 301*200

problem_str = 'bc'
title = 'branin-Currin'

log_phv = True
log_func = np.log10

plot_list = []
legend_list = []
for mth in mths:
    result = []
    dir_path = 'logs/mo_benchmark_%s_%d/%s/' % (problem_str, max_runs, mth)
    for file in os.listdir(dir_path):
        if file.startswith('benchmark_%s_' % (mth)) and file.endswith('.pkl'):
            with open(os.path.join(dir_path, file), 'rb') as f:
                save_item = pkl.load(f)
                hv_diffs, pf, data = save_item
            if recalc == 1:
                if mth.startswith('gpflowopt'):
                    y = data[1]
                else:   # todo
                    y = np.array(list(data.values()))
                hv_diffs = []
                history_container = MOHistoryContainer(None)
                for i in range(y.shape[0]):
                    history_container.add(i, y[i].tolist())
                    pf = history_container.get_pareto_front()   # avoid greater than referencePoint
                    hv = Hypervolume(referencePoint).compute(pf)
                    hv_diff = real_hv - hv
                    hv_diffs.append(hv_diff)
            elif recalc == 2:   # only calculate final result
                if mth.startswith('gpflowopt'):
                    y = data[1]
                else:   # todo
                    y = np.array(list(data.values()))
                history_container = MOHistoryContainer(None)
                for i in range(y.shape[0]):
                    history_container.add(i, y[i].tolist())
                pf = history_container.get_pareto_front()
                hv = Hypervolume(referencePoint).compute(pf)
                hv_diff = real_hv - hv
                hv_diffs = [hv_diff] * max_runs
            if len(hv_diffs) != max_runs:
                print('Error len: ', file, len(hv_diffs))
                continue
            result.append(hv_diffs)
            print('last hv_diff =', hv_diffs[-1])
    print('result rep=', len(result), mth)
    if log_phv:
        result = log_func(result)  # log
    mean_res = np.mean(result, axis=0)
    std_res = np.std(result, axis=0)

    # plot
    x = np.arange(len(mean_res)) + 1
    # p, = plt.plot(x, mean_res)
    p = plt.errorbar(x, mean_res, yerr=std_res*0.5, fmt='', capthick=0.5, capsize=3, errorevery=max_runs//10)
    plot_list.append(p)
    legend_list.append(mth)
    print('last mean,std:', mean_res[-1], std_res[-1])

plt.legend(plot_list, legend_list, loc='upper right', fontsize=12)
plt.title(title, fontsize=18)
plt.xlabel('Iteration', fontsize=15)
if log_phv:
    plt.ylabel('Log Hypervolume Difference', fontsize=15)
else:
    plt.ylabel('Hypervolume Difference', fontsize=15)
plt.show()
