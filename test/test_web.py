import time
import numpy as np

from litebo.artifact.remote_advisor import RemoteAdvisor
from litebo.utils.constants import SUCCESS
from litebo.utils.config_space import Configuration, ConfigurationSpace, UniformFloatHyperparameter


def townsend(config):
    X = np.array(list(config.get_dictionary().values()))
    res = dict()

    res['objs'] = [-(np.cos((X[0]-0.1)*X[1])**2 + X[0] * np.sin(3*X[0]+X[1]))]
    res['constraints'] = [-(-np.cos(1.5*X[0]+np.pi)*np.cos(1.5*X[1])+np.sin(1.5*X[0]+np.pi)*np.sin(1.5*X[1]))]
    return res


# Send task id and config space at register
task_id = time.time()
townsend_params = {
    'float': {
        'x1': (-2.25, 2.5, 0),
        'x2': (-2.5, 1.75, 0)
    }
}
townsend_cs = ConfigurationSpace()
townsend_cs.add_hyperparameters([UniformFloatHyperparameter(e, *townsend_params['float'][e]) for e in townsend_params['float']])

# Create remote advisor
config_advisor = RemoteAdvisor(townsend_cs, '127.0.0.1', 8001, num_constraints=1, random_state=1)

# Simulate 50 iterations
for _ in range(50):
    config_dict = config_advisor.get_suggestion()
    config = Configuration(config_advisor.config_space, config_dict)
    obs = townsend(config)
    config_advisor.update_observation(config_dict, obs['objs'], obs['constraints'], trial_state=SUCCESS)

incumbents, history = config_advisor.get_result()
print(incumbents)
