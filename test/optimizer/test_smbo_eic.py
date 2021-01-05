import numpy as np

import os
import sys
sys.path.insert(0, os.getcwd())

from litebo.optimizer.generic_smbo import SMBO
from litebo.utils.config_space import ConfigurationSpace, UniformFloatHyperparameter


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
# X = np.array([[ 0.91666667, -1.08333333],
#               [-0.66666667,  0.33333333],
#               [-1.19444444, -1.55555556],
#               [ 1.44444444,  0.80555556],
#               [ 0.38888889, -2.5       ],
#               [-2.25      , -0.13888889],
#               [ 2.5       , -0.61111111],
#               [-0.13888889,  1.75      ],
#               [ 1.97222222, -2.02777778],
#               [-1.72222222,  1.27777778]])
# townsend_initial_configs = [Configuration(townsend_cs, {'x1': X[i, 0], 'x2': X[i, 1]}) for i in range(X.shape[0])]

bo = SMBO(townsend, townsend_cs,
          num_constraints=1,
          # initial_configurations=townsend_initial_configs,
          acq_optimizer_type='random_scipy',
          max_runs=60)
bo.run()
