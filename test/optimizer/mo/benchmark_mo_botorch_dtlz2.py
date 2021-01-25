"""
example cmdline:

python test/optimizer/mo/benchmark_mo_botorch_dtlz2.py --n 110 --x 12 --y 2 --q 1 --rep 1 --start_id 0

"""
import os
import sys
import time
import numpy as np
import argparse
import pickle as pkl

import torch

sys.path.insert(0, '../botorch/')   # for dev

sys.path.insert(0, os.getcwd())
from test.test_utils import timeit, seeds
from litebo.utils.multi_objective import Hypervolume

parser = argparse.ArgumentParser()
parser.add_argument('--n', type=int, default=110)
parser.add_argument('--x', type=int, default=12)
parser.add_argument('--y', type=int, default=2)
parser.add_argument('--q', type=int, default=1)
parser.add_argument('--rep', type=int, default=1)
parser.add_argument('--start_id', type=int, default=0)

args = parser.parse_args()
max_runs = args.n
num_inputs = args.x
num_objs = args.y
BATCH_SIZE = args.q     # parallel q
rep = args.rep
start_id = args.start_id
mth = 'botorch-qehvi'

# set problem
from litebo.benchmark.objective_functions.synthetic import DTLZ2
problem = DTLZ2(num_inputs, num_objs, random_state=None)
problem_str = 'DTLZ2-%d-%d' % (num_inputs, num_objs)
problem.ref_point = -1 * torch.tensor(problem.ref_point, dtype=torch.float)  # caution
problem.bounds = torch.tensor(problem.bounds, dtype=torch.float).transpose(-1, -2)  # caution


tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}


from botorch.models.gp_regression import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch.utils.transforms import unnormalize
from botorch.utils.sampling import draw_sobol_samples


def generate_initial_data(n):
    # generate training data
    train_x = draw_sobol_samples(bounds=problem.bounds, n=1, q=n, seed=torch.randint(1000000, (1,)).item()).squeeze(0)
    train_x = train_x.numpy().astype(np.float64)    # caution
    train_obj = -1 * np.vstack([problem(tx, convert=False)['objs'] for tx in train_x]).astype(np.float64)    # caution
    train_x, train_obj = torch.from_numpy(train_x), torch.from_numpy(train_obj)
    return train_x, train_obj


def initialize_model(train_x, train_obj):
    # define models for objective and constraint
    model = SingleTaskGP(train_x, train_obj, outcome_transform=Standardize(m=train_obj.shape[-1]))
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    return mll, model


from botorch.optim.optimize import optimize_acqf, optimize_acqf_list
from botorch.acquisition.objective import GenericMCObjective
from botorch.utils.multi_objective.scalarization import get_chebyshev_scalarization
from botorch.utils.multi_objective.box_decomposition import NondominatedPartitioning
from botorch.acquisition.multi_objective.monte_carlo import qExpectedHypervolumeImprovement
from botorch.utils.sampling import sample_simplex

standard_bounds = torch.zeros(2, problem.dim, **tkwargs)
standard_bounds[1] = 1


def optimize_qehvi_and_get_observation(model, train_obj, sampler):
    """Optimizes the qEHVI acquisition function, and returns a new candidate and observation."""
    # partition non-dominated space into disjoint rectangles
    partitioning = NondominatedPartitioning(num_outcomes=problem.num_objs, Y=train_obj)
    acq_func = qExpectedHypervolumeImprovement(
        model=model,
        ref_point=problem.ref_point.tolist(),  # use known reference point
        partitioning=partitioning,
        sampler=sampler,
    )
    # optimize
    candidates, _ = optimize_acqf(
        acq_function=acq_func,
        bounds=standard_bounds,
        q=BATCH_SIZE,
        num_restarts=20,
        raw_samples=1024,  # used for intialization heuristic
        options={"batch_limit": 5, "maxiter": 200, "nonnegative": True},
        sequential=True,
    )
    # observe new values
    new_x = unnormalize(candidates.detach(), bounds=problem.bounds)
    new_x = new_x.numpy().astype(np.float64)    # caution
    print(f'evaluate {new_x.shape[0]} configs on real objective')
    new_obj = -1 * np.vstack([problem(tx, convert=False)['objs'] for tx in new_x]).astype(np.float64)    # caution
    new_x, new_obj = torch.from_numpy(new_x), torch.from_numpy(new_obj)
    return new_x, new_obj


from botorch import fit_gpytorch_model
from botorch.acquisition.monte_carlo import qExpectedImprovement, qNoisyExpectedImprovement
from botorch.sampling.samplers import SobolQMCNormalSampler
from botorch.exceptions import BadInitialCandidatesWarning
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.utils.multi_objective.hypervolume import Hypervolume
import time

import warnings

warnings.filterwarnings('ignore', category=BadInitialCandidatesWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

init_num = 6
MC_SAMPLES = 128

hv = Hypervolume(ref_point=problem.ref_point)

with timeit('%s all' % (mth,)):
    for run_i in range(start_id, start_id + rep):
        seed = seeds[run_i]
        np.random.seed(seed)
        torch.manual_seed(seed)
        with timeit('%s %d %d' % (mth, run_i, seed)):
            hvs_qehvi= []
            hv_diffs = []

            # call helper functions to generate initial training data and initialize model
            train_x_qehvi, train_obj_qehvi = generate_initial_data(n=init_num)
            mll_qehvi, model_qehvi = initialize_model(train_x_qehvi, train_obj_qehvi)
            X_init = train_x_qehvi.numpy()
            Y_init = -1 * train_obj_qehvi.numpy()   # for plot

            # calculate hypervolume of init data
            for i in range(init_num):
                train_obj_i = train_obj_qehvi[:init_num+1]
                # compute pareto front
                pareto_mask = is_non_dominated(train_obj_i)
                pareto_y = train_obj_i[pareto_mask]
                # compute hypervolume
                volume = hv.compute(pareto_y)
                hvs_qehvi.append(volume)
                hv_diff = problem.max_hv - volume
                hv_diffs.append(hv_diff)

            # run (max_runs - init_num) rounds of BayesOpt after the initial random batch
            for iteration in range(init_num + 1, max_runs + 1):
                print('\n===start iter', iteration)

                t0 = time.time()

                # fit the models
                fit_gpytorch_model(mll_qehvi)

                # define the qEI and qNEI acquisition modules using a QMC sampler
                qehvi_sampler = SobolQMCNormalSampler(num_samples=MC_SAMPLES)

                # optimize acquisition functions and get new observations
                new_x_qehvi, new_obj_qehvi = optimize_qehvi_and_get_observation(
                    model_qehvi, train_obj_qehvi, qehvi_sampler
                )

                # update training points
                train_x_qehvi = torch.cat([train_x_qehvi, new_x_qehvi])
                train_obj_qehvi = torch.cat([train_obj_qehvi, new_obj_qehvi])

                # update progress
                # compute pareto front
                pareto_mask = is_non_dominated(train_obj_qehvi)
                pareto_y = train_obj_qehvi[pareto_mask]
                # compute hypervolume
                volume = hv.compute(pareto_y)
                hvs_qehvi.append(volume)
                hv_diff = problem.max_hv - volume
                hv_diffs.append(hv_diff)

                # reinitialize the models so they are ready for fitting on next iteration
                # Note: we find improved performance from not warm starting the model hyperparameters
                # using the hyperparameters from the previous iteration
                mll_qehvi, model_qehvi = initialize_model(train_x_qehvi, train_obj_qehvi)

                t1 = time.time()

                print(
                    f"Batch {iteration:>2}: Hypervolume (qEHVI) = "
                    f"({hvs_qehvi[-1]:>4.2f}), "
                    f"time = {t1 - t0:>4.2f}."
                )
                print('pareto num =', pareto_y.shape[0])

            # Save result
            pareto_mask = is_non_dominated(train_obj_qehvi)
            pareto_y = train_obj_qehvi[pareto_mask]
            pf = -1 * pareto_y.numpy()
            print(seed, mth, 'pareto num:', pf.shape[0])
            print(seed, 'real hv =', problem.max_hv)
            print(seed, 'hv_diffs:', hv_diffs)

            X = train_x_qehvi.numpy()
            Y = -1 * train_obj_qehvi.numpy()
            data = (X, Y)

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
    pareto_mask = is_non_dominated(train_obj_qehvi)
    pareto_y = train_obj_qehvi[pareto_mask]
    pf = -1 * pareto_y.numpy()

    plt.scatter(pf[:, 0], pf[:, 1], label=mth)
    plt.scatter(Y_init[:, 0], Y_init[:, 1], label='init', marker='x')
    plt.title('Pareto Front of DTLZ2-%d-%d' % (num_inputs, num_objs))
    plt.xlabel('Objective 1')
    plt.ylabel('Objective 2')
    plt.legend()
    plt.show()
