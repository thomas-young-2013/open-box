import numpy as np
import pickle

import os
import sys
sys.path.insert(0, os.getcwd())

from litebo.optimizer.parallel_smbo import pSMBO
from litebo.config_space import ConfigurationSpace, Configuration, \
    UniformFloatHyperparameter, UniformIntegerHyperparameter


# Unconstrained Ackley problem
def ackley(config):
    """
    X is a d-dimensional vector.
    We take d = 10.
    """
    res = dict()
    X = np.array(list(config.get_dictionary().values()))

    a = 20
    b = 0.2
    c = 2*np.pi
    d = 10
    s1 = -a*np.exp(-b*np.sqrt(1/d*(X**2).sum()))
    s2 = -np.exp(1/d*np.cos(c*X).sum())
    s3 = a + np.exp(1)
    res['objs'] = [s1 + s2 + s3]
    res['constraints'] = []
    return res

ackley_params = {
    'float': {f'x{i}': (-10, 15, 2.5) for i in range(1, 11)}
}
ackley_cs = ConfigurationSpace()
ackley_cs.add_hyperparameters([UniformFloatHyperparameter(e, *ackley_params['float'][e]) for e in ackley_params['float']])

# TODO Good initial design for Constrained Ackley problem
def c_ackley(config):
    res = dict()
    X = np.array(list(config.get_dictionary().values()))

    a = 20
    b = 0.2
    c = 2*np.pi
    d = 10
    s1 = -a*np.exp(-b*np.sqrt(1/d*(X**2).sum()))
    s2 = -np.exp(1/d*np.cos(c*X).sum())
    s3 = a + np.exp(1)
    res['objs'] = [s1 + s2 + s3]
    res['constraints'] = [X.sum(),
                          (X**2).sum() - 25]
    return res

c_ackley_params = {
    'float': {f'x{i}': (-5, 10, 2.5) for i in range(1, 11)}
}
c_ackley_cs = ConfigurationSpace()
c_ackley_cs.add_hyperparameters([UniformFloatHyperparameter(e, *c_ackley_params['float'][e]) for e in c_ackley_params['float']])


# TR
# bo = pSMBO(ackley, ackley_cs,
#            batch_size=4,
#            parallel_strategy='ts',
#            init_strategy='sobol',
#            use_trust_region=True,
#            initial_runs=20,  # Set initial runs to 2*dim
#            max_runs=250)
# bo.run()

# EI
# bo = pSMBO(ackley, ackley_cs,
#            batch_size=4,
#            parallel_strategy='ts',
#            init_strategy='sobol',
#            use_trust_region=True,
#            initial_runs=20,  # Set initial runs to 2*dim
#            max_runs=250)
# bo.run()

# SCBO
bo = pSMBO(c_ackley, c_ackley_cs,
           num_constraints=2,
           batch_size=4,
           parallel_strategy='ts',
           init_strategy='latin_hypercube',
           options={'lh_criterion': None},
           use_trust_region=True,
           initial_runs=20,  # Set initial runs to 2*dim
           max_runs=250)
bo.run()

# Save results
with open('bo.pkl', 'wb') as f:
    pickle.dump(bo.get_history(), f, protocol=pickle.HIGHEST_PROTOCOL)
