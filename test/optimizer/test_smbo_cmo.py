import os
import sys
import numpy as np
sys.path.insert(0, os.getcwd())

from litebo.optimizer.generic_smbo import SMBO
from litebo.utils.config_space import ConfigurationSpace, UniformFloatHyperparameter


def branin_currin(config):
    dic = config.get_dictionary()
    x1 = dic.get('x1')
    x2 = dic.get('x2')
    px1 = 15*x1 - 5
    px2 = 15*x2
    res = dict()

    f1 = (px2 - 5.1/(4*np.pi**2) * px1**2 + 5/np.pi * px1 - 6)**2 + 10 * (1 - 1/(8*np.pi)) * np.cos(px1) + 10
    f2 = (1 - np.exp(-1/(2*x2))) * (2300 * x1**3 + 1900 * x1**2 + 2092 * x1 + 60) / (100 * x1**3 + 500 * x1**2 + 4*x1 + 20)
    res['objs'] = [f1, f2]
    # res['constraints'] = [(px1 - 2.5)**2 + (px2 - 7.5)**2 - 50]
    res['constraints'] = []
    return res

bc_params = {
    'float': {
        'x1': (0, 1, 0.5),
        'x2': (0, 1, 0.5)
    }
}
bc_cs = ConfigurationSpace()
bc_cs.add_hyperparameters([UniformFloatHyperparameter(e, *bc_params['float'][e]) for e in bc_params['float']])
bc_max_hv = 59.36011874867746
bc_ref_point = [18., 6.]


def c2dtlz2(config):
    """
    Synthetic C2-DTLZ2 test.
    d = 12 parameters, M = 2 objectives, V = 1 constraint.
    """
    X = np.array(list(config.get_dictionary().values()))
    res = dict()

    M = 2
    r = 0.2

    g = np.sum((X[M-1:] - 0.5)**2)
    f1 = (1+g)*np.cos(np.pi/2*X[0])
    f2 = (1+g)*np.sin(np.pi/2*X[0])
    res['objs'] = [f1, f2]
    m1 = (f1 - 1)**2 + f2**2 - r**2
    m2 = (f2 - 1)**2 + f1**2 - r**2
    m3 = (f1 - 1/np.sqrt(M))**2 + (f2 - 1/np.sqrt(M))**2 - 2*r**2
    res['constraints'] = [np.min([m1, m2, m3])]
    return res

c2dtlz2_params = {
    'float': {
        'x1': (-2.25, 2.5, 0),
        'x2': (-2.5, 1.75, 0)
    }
}
c2dtlz2_cs = ConfigurationSpace()
c2dtlz2_cs.add_hyperparameters([UniformFloatHyperparameter(e, *c2dtlz2_params['float'][e]) for e in c2dtlz2_params['float']])
c2dtlz2_max_hv = 0.3996406303723544
c2dtlz2_ref_point = [1.1, 1.1]


bo = SMBO(branin_currin, bc_cs,
          num_objs=2,
          ref_point=[100, 30],
          max_runs=100)
bo.run()

hvs = bo.get_history().hv_data
log_hv_diff = np.log10(bc_max_hv - np.asarray(hvs))
