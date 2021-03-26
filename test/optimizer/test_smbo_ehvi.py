import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.getcwd())

from litebo.optimizer.generic_smbo import SMBO
from litebo.benchmark.objective_functions.synthetic import VehicleSafety

num_inputs = 5
num_objs = 3
acq_optimizer_type = 'random_scipy'
seed = 1
prob = VehicleSafety()
initial_runs = 2 * (num_inputs + 1)
max_runs = 100 + initial_runs

bo = SMBO(prob.evaluate, prob.config_space,
          task_id='ehvi',
          num_objs=prob.num_objs,
          num_constraints=prob.num_constraints,
          acq_type='ehvi',
          acq_optimizer_type=acq_optimizer_type,
          surrogate_type='gp',
          ref_point=prob.ref_point,
          max_runs=max_runs,
          initial_runs=initial_runs,
          init_strategy='sobol',
          random_state=seed)
bo.run()

hvs = bo.get_history().hv_data

pf = np.asarray(bo.get_history().get_pareto_front())
if pf.shape[-1] == 2:
    plt.scatter(pf[:, 0], pf[:, 1])
elif pf.shape[-1] == 3:
    ax = plt.axes(projection='3d')
    ax.scatter3D(pf[:, 0], pf[:, 1], pf[:, 2])
plt.show()

try:
    log_hv_diff = np.log10(prob.max_hv - np.asarray(hvs))[initial_runs:]
    plt.plot(log_hv_diff)
except NotImplementedError:
    plt.plot(hvs)
plt.show()
