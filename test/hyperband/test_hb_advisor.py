import time
import numpy as np

import sys
sys.path.insert(0, '.')
from litebo.core.hyperband_advisor import HyperbandAdvisor
from litebo.utils.constants import MAXINT, SUCCESS, FAILED, TIMEOUT
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, Constant, UniformIntegerHyperparameter


def get_cs():
    cs = ConfigurationSpace()
    n_estimators = UniformFloatHyperparameter("n_estimators", 100, 1000, default_value=500, q=50)
    num_leaves = UniformIntegerHyperparameter("num_leaves", 31, 2047, default_value=128)
    learning_rate = UniformFloatHyperparameter("learning_rate", 1e-3, 0.3, default_value=0.1, log=True)
    cs.add_hyperparameters([n_estimators, num_leaves, learning_rate])
    return cs


def obj_func(config, n_iter, total_iter, extra_info):
    #print('===obj called:', config, n_iter, total_iter, extra_info)
    # time.sleep(np.random.randint(5))  # todo for parallel test
    params = config.get_dictionary()

    n_estimators = params['n_estimators']
    num_leaves = params['num_leaves']
    learning_rate = params['learning_rate']
    perf = n_estimators + num_leaves + learning_rate + np.random.rand() / 10000
    return perf


R = 27
eta = 3

inner_iter = 10

advisor = HyperbandAdvisor(get_cs(), R=R, eta=eta, restart_needed=True, skip_last=0)

for i in range(inner_iter):
    suggestion_list = []
    while True:
        suggestion = advisor.get_mf_suggestion()
        if suggestion is None:
            break
        suggestion_list.append(suggestion)

    for idx, suggestion in enumerate(suggestion_list):
        config, n_iter, total_iter, extra_info = suggestion
        perf = obj_func(*suggestion)
        observation = (config, perf, SUCCESS)
        print('update (disorderly)', idx+1, perf)
        advisor.update_mf_observation(observation)

for i, obj in enumerate(advisor.incumbent_perfs):
    print('%d-th config: %s, obj: %f.' % (i + 1, str(advisor.incumbent_configs[i]), advisor.incumbent_perfs[i]))
