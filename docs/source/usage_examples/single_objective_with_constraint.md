# Single Objective with Constraint

```python
import numpy as np
from litebo.optimizer.generic_smbo import SMBO
from litebo.utils.config_space import ConfigurationSpace, UniformFloatHyperparameter


def townsend(config):
    config_dict = config.get_dictionary()
    X = np.array([config_dict['x%d' % i] for i in range(2)])
    res = dict()
    res['objs'] = (-(np.cos((X[0]-0.1)*X[1])**2 + X[0] * np.sin(3*X[0]+X[1])), )
    res['constraints'] = (-(-np.cos(1.5*X[0]+np.pi)*np.cos(1.5*X[1])+np.sin(1.5*X[0]+np.pi)*np.sin(1.5*X[1])), )
    return res


townsend_params = {
    'float': {
        'x0': (-2.25, 2.5, 0),
        'x1': (-2.5, 1.75, 0)
    }
}
townsend_cs = ConfigurationSpace()
townsend_cs.add_hyperparameters([UniformFloatHyperparameter(name, *para)
                                 for name, para in townsend_params['float'].items()])

bo = SMBO(townsend, townsend_cs,
          num_constraints=1,
          num_objs=1,
          acq_optimizer_type='random_scipy',
          max_runs=100,
          task_id='soc')
bo.run()
print(bo.get_history())
```
