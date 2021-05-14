import datetime
import hashlib
import time
import numpy as np

from litebo.artifact.remote_advisor import RemoteAdvisor
from litebo.utils.constants import SUCCESS
from litebo.utils.config_space import Configuration, ConfigurationSpace, UniformFloatHyperparameter


def townsend(config):
    X = np.array(list(config.get_dictionary().values()))
    res = dict()
    time.sleep(3)
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
pwd = '111111'
pwd = hashlib.md5(pwd.encode(encoding='utf-8')).hexdigest()
user_email = '2322171400@qq.com'
server = '39.101.191.37'
server = '127.0.0.1'

port = 11425
port = 8001
config_advisor = RemoteAdvisor(townsend_cs, server, port, user_email,pwd,task_name="task_test")

# Simulate 50 iterations
for _ in range(200):

    config_dict = config_advisor.get_suggestion()
    config = Configuration(config_advisor.config_space, config_dict)
    trial_info = {}
    start_time = datetime.datetime.now()
    obs = townsend(config)

    trial_info['cost'] = (datetime.datetime.now()- start_time).seconds
    trial_info['worker_id'] = 0
    trial_info['trial_info'] = 'None'
    config_advisor.update_observation(config_dict, obs['objs'], obs['constraints'], trial_info= trial_info,trial_state=SUCCESS)

incumbents, history = config_advisor.get_result()
print(incumbents)
