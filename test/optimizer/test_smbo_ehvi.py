import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.getcwd())

from litebo.optimizer.generic_smbo import SMBO
from litebo.benchmark.objective_functions.synthetic import DTLZ2


prob = DTLZ2()
bo = SMBO(prob.evaluate, prob.config_space,
          task_id='ehvi',
          num_objs=prob.num_objs,
          num_constraints=prob.num_constraints,
          acq_type='ehvi',
          acq_optimizer_type='random_scipy',
          surrogate_type='gp',
          ref_point=prob.ref_point,
          max_runs=100,
          random_state=1)
bo.run()

hvs = bo.get_history().hv_data
log_hv_diff = np.log10(prob.max_hv - np.asarray(hvs))

pf = np.asarray(bo.get_history().get_pareto_front())
plt.scatter(pf[:, 0], pf[:, 1])
plt.plot(log_hv_diff)
plt.show()
