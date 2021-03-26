# Parallel Evaluation

Most proposed Bayesian optimization (BO) approaches only allow the exploration of the parameter space to occur 
sequentially. To fully utilize the computing resources in a parallel infrastructure, **OpenBox** provide a 
mechanism for distributed parallelization, where multiple configurations can be evaluated concurrently across workers. 

Two parallel settings are considered:

![An illustration of the sync (left) and async (right) parallel methods using three workers](../assets/parallel_bo.pdf)

1) **Synchronous parallel setting (left)**. The worker pulls new configuration from suggestion server to evaluate until all 
the workers have finished their last evaluations.

2) **Asynchronous parallel setting (right)**. The worker pulls a new configuration when the previous evaluation is completed.

**OpenBox** proposes a local penalization based parallelization mechanism, the goal of which is to sample new 
configurations that are promising and far enough from the configurations being evaluated by other workers.
This mechanism can handle the well-celebrated exploration vs. exploitation trade-off, and meanwhile prevent workers 
from exploring similar configurations.

In this tutorial, we illustrate how to optimize a problem in parallel manner on your local machine with **OpenBox**.

## Problem Setup

First, **define configuration space** to search and **define objective function**
to <font color=#FF0000>**minimize**</font>. Here we use the **Branin** function.

```python
import numpy as np
from litebo.utils.config_space import ConfigurationSpace, UniformFloatHyperparameter

# Define Configuration Space
config_space = ConfigurationSpace()
x1 = UniformFloatHyperparameter("x1", -5, 10, default_value=0)
x2 = UniformFloatHyperparameter("x2", 0, 15, default_value=0)
config_space.add_hyperparameters([x1, x2])

# Define Objective Function
def branin(config):
    config_dict = config.get_dictionary()
    x1 = config_dict['x1']
    x2 = config_dict['x2']

    a = 1.
    b = 5.1 / (4. * np.pi ** 2)
    c = 5. / np.pi
    r = 6.
    s = 10.
    t = 1. / (8. * np.pi)
    y = a * (x2 - b * x1 ** 2 + c * x1 - r) ** 2 + s * (1 - t) * np.cos(x1) + s

    ret = dict(
        objs=(y, )
    )
    return ret
```

If you are not familiar with setting up problem, please refer to 
the [Quick Example Tutorial](../getting_started/quick_example).

## Parallel Evaluation on Local Machine

This time we use <font color=#FF0000>**pSMBO**</font> to optimize the objective function in parallel manner 
on your local machine.

```python
from litebo.optimizer.parallel_smbo import pSMBO

# Parallel Evaluation on Local Machine
bo = pSMBO(branin,
           config_space,
           parallel_strategy='async',
           batch_size=4,
           batch_strategy='median_imputation',
           num_objs=1,
           num_constraints=0,
           max_runs=100,
           surrogate_type='gp',
           time_limit_per_trial=180,
           task_id='parallel')
bo.run()
```

In addition to **objective_function** and **config_space** parameters being passed to **pSMBO**, 
other parameters are as follows:

+ **parallel_strategy='async' / 'sync'** sets whether parallel evaluation is in asynchronous or synchronous mode.
We suggest using **'async'** because it makes better use of resources and achieves better results than **'sync'**.

+ **batch_size=4** sets how many workers are running in parallel.

+ **batch_strategy='median_imputation'** sets the strategy how to make multiple suggestions at the same time.
We implemented several strategies, and we suggest using the default strategy **'median_imputation'** as it supports
more scenarios.

+ **num_objs=1** and **num_constraints=0** indicates our function returns a single objective value with no constraint. 

+ **max_runs=100** means the optimization will take 100 rounds (100 times of objective function evaluation). 

+ **surrogate_type='gp'**. For mathematical problem, we suggest using Gaussian Process (**'gp'**) as Bayesian surrogate
model. For practical problems such as hyperparameter optimization (HPO), we suggest using Random Forest (**'prf'**).

+ **time_limit_per_trial** sets the time budget (seconds) of each objective function evaluation. Once the 
evaluation time exceeds this limit, objective function will return as a failed trial.

+ **task_id** is set to identify the optimization process.

After optimization, call <font color=#FF0000>**print(bo.get_history())**</font> to see the result:

```python
print(bo.get_history())
```

```
+----------------------------------------------+
| Parameters              | Optimal Value      |
+-------------------------+--------------------+
| x1                      | -3.138286          |
| x2                      | 12.292733          |
+-------------------------+--------------------+
| Optimal Objective Value | 0.3985991718620365 |
+-------------------------+--------------------+
| Num Configs             | 100                |
+-------------------------+--------------------+
```
