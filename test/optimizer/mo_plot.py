"""
example cmdline:

python test/optimizer/mo_plot.py --mths mesmo-1,usemo-1,gpflowopt-hvpoi --problem bc --n 110

"""
import os
import argparse
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt

default_mths = 'mesmo-1,usemo-1,gpflowopt-hvpoi'
parser = argparse.ArgumentParser()
parser.add_argument('--n', type=int, default=100)
parser.add_argument('--mths', type=str, default=default_mths)
parser.add_argument('--problem', type=str, default='bc')

args = parser.parse_args()
max_runs = args.n
mths = args.mths.split(',')     # list of 'mth-sample_num'

# set problem
problem_str = args.problem
if problem_str == 'bc':
    title = 'branin-Currin'
    log_phv = True
elif problem_str.startswith('lightgbm'):
    title = problem_str
    log_phv = False
else:
    raise ValueError('Unknown problem:', problem_str)


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
            if len(hv_diffs) != max_runs:
                print('Error len: ', file, len(hv_diffs))
                continue
            result.append(hv_diffs)
            print(hv_diffs[-1])
    print('result rep=', len(result))
    if log_phv:
        result = np.log(result)  # log
    mean_res = np.mean(result, axis=0)
    std_res = np.std(result, axis=0)

    # plot
    x = np.arange(len(mean_res)) + 1
    # p, = plt.plot(mean_res)
    p = plt.errorbar(x, mean_res, yerr=std_res*0.5, fmt='', capthick=0.5, capsize=3, errorevery=max_runs//10)
    plot_list.append(p)
    legend_list.append(mth)
    print(mean_res[-1], std_res[-1])

plt.legend(plot_list, legend_list, loc='upper right')
plt.title(title)
plt.xlabel('Iteration')
if log_phv:
    plt.ylabel('Log Hypervolume Difference')
else:
    plt.ylabel('Hypervolume Difference')
plt.show()
