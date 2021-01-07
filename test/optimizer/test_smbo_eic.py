import numpy as np

from litebo.optimizer.generic_smbo import SMBO
from litebo.utils.config_space import Configuration, ConfigurationSpace, UniformFloatHyperparameter


def townsend(config):
    X = np.array(list(config.get_dictionary().values()))
    res = dict()
    res['objs'] = (-(np.cos((X[0]-0.1)*X[1])**2 + X[0] * np.sin(3*X[0]+X[1])), )
    res['constraints'] = (-(-np.cos(1.5*X[0]+np.pi)*np.cos(1.5*X[1])+np.sin(1.5*X[0]+np.pi)*np.sin(1.5*X[1])), )
    return res

townsend_params = {
    'float': {
        'x1': (-2.25, 2.5, 0),
        'x2': (-2.5, 1.75, 0)
    }
}
townsend_cs = ConfigurationSpace()
townsend_cs.add_hyperparameters([UniformFloatHyperparameter(e, *townsend_params['float'][e]) for e in townsend_params['float']])

bo = SMBO(townsend, townsend_cs,
          num_constraints=1,
          task_id=1,
          initial_runs=10,
          max_runs=30, random_state=1)
bo.run()
