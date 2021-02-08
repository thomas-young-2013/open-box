"""
example cmdline:

python test/reproduction/mo/mo_plot.py --mths litebo,gpflowopt,botorch --problem dtlz2-12-2 --n 200

"""
import os
import argparse
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style(style='whitegrid')

#plt.rc('text', usetex=True)    # todo
# plt.rc('font', **{'size': 16, 'family': 'Helvetica'})

plt.rc('font', size=16.0, family='sans-serif')
plt.rcParams['font.sans-serif'] = "Tahoma"

plt.rcParams['figure.figsize'] = (8.0, 4.5)
#plt.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]    # todo
plt.rcParams["legend.frameon"] = True
plt.rcParams["legend.facecolor"] = 'white'
plt.rcParams["legend.edgecolor"] = 'gray'
plt.rcParams["legend.fontsize"] = 16
label_size = 24


default_mths = 'gpflowopt,botorch,hypermapper,litebo'
parser = argparse.ArgumentParser()
parser.add_argument('--n', type=int, default=200)
parser.add_argument('--mths', type=str, default=default_mths)
parser.add_argument('--problem', type=str, default='dtlz2-12-2')

args = parser.parse_args()
max_runs = args.n
mths = args.mths.split(',')


def fetch_color_marker(m_list):
    color_dict = dict()
    marker_dict = dict()
    color_list = ['purple', 'royalblue', 'green', 'brown', 'red', 'orange', 'yellowgreen', 'black', 'yellow']
    markers = ['s', '^', '*', 'v', 'o', 'p', '2', 'x', 'd']

    def fill_values(name, idx):
        color_dict[name] = color_list[idx]
        marker_dict[name] = markers[idx]

    for name in m_list:
        if name.startswith('cmaes'):
            fill_values(name, 0)
        elif name.startswith('random-n1'):
            fill_values(name, 1)
        elif name.startswith('random-n2'):
            fill_values(name, 2)
        elif name.startswith('litebo'):
            fill_values(name, 4)
            if 'mesmo' in name:     # todo
                fill_values(name, 3)
        elif name.startswith('smac'):
            fill_values(name, 3)
        elif name.startswith('hyperopt'):
            fill_values(name, 5)
        elif name.startswith('gpflowopt'):
            fill_values(name, 6)
        elif name.startswith('botorch'):
            fill_values(name, 7)
        else:
            print(name)
            fill_values(name, 8)
    return color_dict, marker_dict


def get_mth_legend(mth):
    mth = mth.lower()
    if mth == 'gpflowopt':
        return 'GPflowOpt'
    elif mth == 'random-n1':
        return 'Random'
    elif mth == 'random-n2':
        return '2$\\times$Random'
    elif mth == 'cmaes':
        return 'CMA-ES'
    elif mth == 'litebo':
        return 'Open-BOX'
    elif mth == 'smac':
        return 'SMAC3'
    elif mth == 'hyperopt':
        return 'Hyperopt'
    elif mth == 'botorch':
        return 'Botorch'
    elif mth == 'hypermapper':
        return 'Hypermapper'
    else:
        return mth


# set problem
problem_str = args.problem
title = problem_str
log_phv = True
log_func = np.log10
# plt.ylim(11.9, 12.45)
# plt.xlim(0, 201)
# plt.subplots_adjust(top=0.97, right=0.968, left=0.11, bottom=0.14)
std_scale = 1.0
if problem_str.startswith('dtlz2'):
    pass
else:
    print('Unknown problem: %s. Use default plot settings.' % (problem_str,))
print(f'title={title}, log_phv={log_phv}, log_func={log_func}, std_scale={std_scale}.')

color_dict, marker_dict = fetch_color_marker(mths)

plot_list = []
legend_list = []
for mth in mths:
    if mth.startswith('random'):
        pos = mth.find('-n') + 2
        times = int(mth[pos:])
    else:
        times = 1
    result = []
    dir_path = 'logs/mo_benchmark_%s_%d/%s/' % (problem_str, max_runs, mth)
    for file in os.listdir(dir_path):
        if file.startswith('benchmark_%s_' % (mth,)) and file.endswith('.pkl'):
            with open(os.path.join(dir_path, file), 'rb') as f:
                save_item = pkl.load(f)
                if mth == 'gpflowopt' or mth == 'botorch':
                    hv_diffs, pf, X, Y, time_list = save_item
                else:
                    hv_diffs, pf, config_list, perf_list, time_list = save_item
            if len(hv_diffs) / times != max_runs:
                print('Error len: ', file, len(hv_diffs))
                continue
            result.append(hv_diffs)
            print('last hv_diff =', hv_diffs[-1])
            if pf is not None:
                print(mth, 'pareto num =', len(pf))
    print('result rep =', len(result), mth)
    if log_phv:
        result = log_func(result)  # log
    mean_res = np.mean(result, axis=0)
    std_res = np.std(result, axis=0)

    # plot
    x = (np.arange(len(mean_res)) + 1) / times
    p, = plt.plot(x, mean_res, label=get_mth_legend(mth), color=color_dict[mth])
    # p = plt.errorbar(x, mean_res, yerr=std_res*std_scale, fmt='', capthick=0.5, capsize=3, errorevery=max_runs//10)
    plt.fill_between(x, mean_res - std_res*std_scale, mean_res + std_res*std_scale, alpha=0.2, facecolor=color_dict[mth])
    plot_list.append(p)
    legend_list.append(get_mth_legend(mth))
    print('last mean,std:', mean_res[-1], std_res[-1])

plt.legend()
# plt.title(title, fontsize=18)
# todo
# plt.xlabel('\\textbf{Iteration}', fontsize=label_size)
# if log_obj:
#     plt.ylabel('\\textbf{Log Hypervolume Difference}', fontsize=label_size)
# else:
#     plt.ylabel('\\textbf{Hypervolume Difference}', fontsize=label_size)
# plt.savefig('math_%s.pdf' % problem_str)

plt.show()
