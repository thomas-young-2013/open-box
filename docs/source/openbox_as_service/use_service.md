# Use OpenBox Service

This tutorial helps you use a remote **OpenBox** service.

## Register an Account

Visit <http://127.0.0.1:11425/user_board/index/> (replace "127.0.0.1:11425" by service ip:port) and you will see
the homepage of **OpenBox** service. Register an account by email to use the service.

You need to activate your account by clicking on the link in the activation email.

## Submit a Task

Here is an example of how to use <font color=#FF0000>**RemoteAdvisor**</font> to interact with **OpenBox** service.

```python
import datetime
import time
import numpy as np

from openbox.artifact.remote_advisor import RemoteAdvisor
from openbox.utils.constants import SUCCESS, FAILED, TIMEOUT, MEMOUT
from openbox.utils.config_space import Configuration, ConfigurationSpace, UniformFloatHyperparameter


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
townsend_cs.add_hyperparameters([UniformFloatHyperparameter(e, *townsend_params['float'][e])
                                 for e in townsend_params['float']])

max_runs = 50
# Create remote advisor
config_advisor = RemoteAdvisor(config_space=townsend_cs,
                               server_ip='127.0.0.1',
                               port=11425,
                               email='your_email@xxxx.com',
                               password='your_password',
                               num_constraints=1,
                               max_runs=max_runs,
                               task_name="task_test",
                               task_id=task_id)

# Simulate max_runs iterations
for idx in range(max_runs):

    config_dict = config_advisor.get_suggestion()
    config = Configuration(config_advisor.config_space, config_dict)
    print('Get %d config: %s' % (idx+1, config))
    trial_info = {}
    start_time = datetime.datetime.now()
    obs = townsend(config)

    trial_info['cost'] = (datetime.datetime.now() - start_time).seconds
    trial_info['worker_id'] = 0
    trial_info['trial_info'] = 'None'
    print('Result %d is %s. Update observation to server.' % (idx+1, obs))
    config_advisor.update_observation(config_dict, obs['objs'], obs['constraints'],
                                      trial_info=trial_info, trial_state=SUCCESS)

incumbents, history = config_advisor.get_result()
print(incumbents)
```

+ Remember to set **server_ip, port** of the service and **email, password** of your account when creating 
**RemoteAdvisor**. A task is then registered to the service.

+ Once you create a task, you can get configuration suggestion from the service by calling
<font color=#FF0000>**RemoteAdvisor.get_suggestion()**</font>. 

+ Run your job locally and send results back to the service by calling 
<font color=#FF0000>**RemoteAdvisor.update_observation()**</font>. 

+ Repeat the **get_suggestion** and **update_observation** process to complete the optimization.

If you are not familiar with setting up problem, please refer to 
the [Quick Start Tutorial](../quick_start/quick_start).

## Monitor task on the Web Page

You can always monitor your task and see optimization result on **OpenBox** service web page.

Visit <http://127.0.0.1:11425/user_board/index/> (replace "127.0.0.1:11425" by service ip:port)
and login your account.

You will see all the tasks you created. Click the buttons to further observe the results and manage your tasks.

![](../assets/user_board_example.png)

