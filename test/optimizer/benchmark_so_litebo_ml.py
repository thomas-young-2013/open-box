"""
example cmdline:

python test/optimizer/so/benchmark_so_litebo_math.py --problem branin --n 200 --init 3 --surrogate gp --optimizer scipy --rep 1 --start_id 0

"""
import os
import sys
import time
import numpy as np
import argparse
import pickle as pkl


sys.path.insert(0, os.getcwd())
from so_benchmark_function import get_problem
from litebo.optimizer.generic_smbo import SMBO
from litebo.test_utils import timeit, seeds
from litebo.utils.constants import MAXINT

parser = argparse.ArgumentParser()
parser.add_argument('--problem', type=str)
parser.add_argument('--n', type=int, default=100)
parser.add_argument('--init', type=int, default=3)
parser.add_argument('--init_strategy', type=str, default='random_explore_first')
parser.add_argument('--surrogate', type=str, default='gp', choices=['gp', 'prf'])
parser.add_argument('--optimizer', type=str, default='scipy', choices=['scipy', 'local'])
parser.add_argument('--rep', type=int, default=1)
parser.add_argument('--start_id', type=int, default=0)


# 解析参数  设置常量
args = parser.parse_args()
problem_str = args.problem
max_runs = args.n
initial_runs = args.init
init_strategy = args.init_strategy
surrogate_type = args.surrogate
if args.optimizer == 'scipy':
    acq_optimizer_type = 'random_scipy'
elif args.optimizer == 'local':
    acq_optimizer_type = 'local_random'
else:
    raise ValueError('Unknown optimizer %s' % args.optimizer)
rep = args.rep
start_id = args.start_id
mth = 'litebo'

problem = get_problem(problem_str)
cs = problem.get_configspace(optimizer='smac')
time_limit_per_trial = 600
task_id = '%s_%s' % (mth, problem_str)


def evaluate(mth, run_i, seed):
    print(mth, run_i, seed, '===== start =====', flush=True)    # litebo 5 2137 ===== start =====

    def objective_function(config):
        y = problem.evaluate_config(config)
        res = dict()
        res['config'] = config
        res['objs'] = (y,)
        res['constraints'] = None
        return res

    bo = SMBO(objective_function, cs,
              surrogate_type='prf',            # default: gp
              acq_optimizer_type='local_random',    # default: random_scipy
              initial_runs=initial_runs,                # default: 3
              init_strategy=init_strategy,              # default: random_explore_first
              max_runs=max_runs,
              time_limit_per_trial=time_limit_per_trial, task_id=task_id, random_state=seed)
    # bo.run()  # 现在不再用run了，而使用iterate， 便于中间操作
    config_list = []
    perf_list = []
    time_list = []
    global_start_time = time.time()
    bo.MIN_PERF = [MAXINT]
    for i in range(max_runs):
        config, trial_state, objs, trial_info = bo.iterate()  # 这里进行训练循环 次数为--n 参数，一次次iterate
        # 同时iterate的时候还会输出[INFO]

        # Visualization: min_bound line of the objective
        for idx, obj in enumerate(objs):
            if obj < bo.MIN_PERF[idx]:
                bo.writer.add_scalar('data/run-%d-objective-%d' % (run_i, idx + 1), obj, bo.iteration_id)
                bo.MIN_PERF[idx] = obj
            else:
                bo.writer.add_scalar('data/run-%d-objective-%d' % (run_i, idx + 1), bo.MIN_PERF[idx], bo.iteration_id)
        # Visualization:  hyperparameter
        config_dict = config.get_dictionary()
        score_dict = {'objective': obj}
        bo.writer.add_hparams(config_dict, score_dict, name="rep-%d-iter-%d" % (run_i, i + 1))

        global_time = time.time() - global_start_time         # 计算从训练开始到现在的时间
        # 终端输出谋面的普通内容
        print(seed, i, objs, config, trial_state, trial_info, 'time=', global_time)
        # 2137 0 (-0.8024522981235505,) Configuration:
        # 一系列 xxx, Value: xxx的行 都是config里面的
        # 0 None time= 0.18564987182617188
        config_list.append(config)
        perf_list.append(objs[0])
        time_list.append(global_time)

    return config_list, perf_list, time_list


with timeit('%s all' % (mth,)):
    for run_i in range(start_id, start_id + rep):
        seed = seeds[run_i]
        with timeit('%s %d %d' % (mth, run_i, seed)):
            # Evaluate
            config_list, perf_list, time_list = evaluate(mth, run_i, seed)

            # Save result
            print('=' * 20)
            print(seed, mth, config_list, perf_list, time_list)
            print(seed, mth, 'best perf', np.min(perf_list))

            timestamp = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
            dir_path = 'logs/so_benchmark_%s_%d/%s/' % (problem_str, max_runs, mth)
            file = 'benchmark_%s_%04d_%s.pkl' % (mth, seed, timestamp)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            with open(os.path.join(dir_path, file), 'wb') as f:
                save_item = (config_list, perf_list, time_list)
                pkl.dump(save_item, f)
            print(dir_path, file, 'saved!', flush=True)
