# Multi-Objective with Constraint

```python
import numpy as np
import matplotlib.pyplot as plt

from litebo.optimizer.generic_smbo import SMBO
from litebo.benchmark.objective_functions.synthetic import CONSTR

prob = CONSTR()
dim = 2
initial_runs = 2 * (dim + 1)

bo = SMBO(prob.evaluate,
          prob.config_space,
          num_objs=prob.num_objs,
          num_constraints=prob.num_constraints,
          acq_type='ehvic',
          acq_optimizer_type='random_scipy',
          surrogate_type='gp',
          ref_point=prob.ref_point,
          max_runs=100,
          initial_runs=initial_runs,
          init_strategy='sobol',
          task_id='moc',
          random_state=1)
bo.run()

# plot pareto front
pareto_front = np.asarray(bo.get_history().get_pareto_front())
if pareto_front.shape[-1] in (2, 3):
    if pareto_front.shape[-1] == 2:
        plt.scatter(pareto_front[:, 0], pareto_front[:, 1])
        plt.xlabel('Objective 1')
        plt.ylabel('Objective 2')
    elif pareto_front.shape[-1] == 3:
        ax = plt.axes(projection='3d')
        ax.scatter3D(pareto_front[:, 0], pareto_front[:, 1], pareto_front[:, 2])
        ax.set_xlabel('Objective 1')
        ax.set_ylabel('Objective 2')
        ax.set_zlabel('Objective 3')
    plt.title('Pareto Front')
    plt.show()

# plot hypervolume
hypervolume = bo.get_history().hv_data
log_hv_diff = np.log10(prob.max_hv - np.asarray(hypervolume))
plt.plot(log_hv_diff)
plt.xlabel('Iteration')
plt.ylabel('Log Hypervolume Difference')
plt.show()
```
