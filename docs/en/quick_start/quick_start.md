# Quick Start

This tutorial helps you run your first example with **OpenBox**.

## Space Definition

First, define a configuration space using the package **ConfigSpace**.

```python
from openbox.utils.config_space import ConfigurationSpace, UniformFloatHyperparameter

# Define Configuration Space
config_space = ConfigurationSpace()
x1 = UniformFloatHyperparameter("x1", -5, 10, default_value=0)
x2 = UniformFloatHyperparameter("x2", 0, 15, default_value=0)
config_space.add_hyperparameters([x1, x2])
```

In this example, we create an empty configuration space, and then add two uniformly distributed float hyperparameters into it.
The first hyperparameter **x1** ranges from -5 to 10, and the second one **x2** ranges from 0 to 15.

The package **ConfigSpace** also supports other types of hyperparameters.
Here are examples of how to define **Integer** and **Categorical** hyperparameters:

```python
from openbox.utils.config_space import UniformIntegerHyperparameter, CategoricalHyperparameter

i = UniformIntegerHyperparameter("i", 0, 100) 
kernel = CategoricalHyperparameter("kernel", ["rbf", "poly", "sigmoid"], default_value="rbf")
```

For advanced usage of **ConfigSpace**, please refer to [ConfigSpace’s documentation](https://automl.github.io/ConfigSpace/master/index.html).

## Objective Definition

Second, define the objective function to be optimized.
Note that **OpenBox** aims to <font color=#FF0000>**minimize**</font> the objective function.
Here we provide an example of the **Branin** function.

```python
import numpy as np

# Define Objective Function
def branin(config):
    x1, x2 = config['x1'], config['x2']
    y = (x2-5.1/(4*np.pi**2)*x1**2+5/np.pi*x1-6)**2+10*(1-1/(8*np.pi))*np.cos(x1)+10
    return y
```

The objective function takes as input a configuration sampled from **ConfigurationSpace**
and outputs the objective value.

## Optimization

After defining the configuration space and the objective function, we can run the optimization process 
as follows:

```python
from openbox.optimizer.generic_smbo import SMBO

# Run Optimization
bo = SMBO(branin,
          config_space,
          num_objs=1,
          num_constraints=0,
          max_runs=50,
          surrogate_type='gp',
          time_limit_per_trial=180,
          task_id='quick_start')
history = bo.run()
```

Here we create a <font color=#FF0000>**SMBO**</font> instance, and pass the objective function **branin** and the 
configuration space **config_space** to it. The other parameters are:

+ **num_objs=1** and **num_constraints=0** indicates our branin function returns a single value with no 
constraint. 

+ **max_runs=50** means the optimization will take 50 rounds (optimizing the objective function 50 times). 

+ **surrogate_type='gp'**. For mathematical problems, we suggest using Gaussian Process (**'gp'**) as Bayesian surrogate
model. For practical problems such as hyperparameter optimization (HPO), we suggest using Random Forest (**'prf'**).

+ **time_limit_per_trial** sets the time budget (seconds) for each objective function evaluation. Once the 
evaluation time exceeds this limit, objective function will return as a failed trial.

+ **task_id** is set to identify the optimization process.

Then, <font color=#FF0000>**bo.run()**</font> is called to start the optimization process.

## Visualization

After the optimization, **bo.run()** returns the optimization history.
Call <font color=#FF0000>**print(history)**</font> to see the result:

```python
print(history)
```

```
+-------------------------+-------------------+
| Parameters              | Optimal Value     |
+-------------------------+-------------------+
| x1                      | -3.138277         |
| x2                      | 12.254526         |
+-------------------------+-------------------+
| Optimal Objective Value | 0.398096578033325 |
+-------------------------+-------------------+
| Num Configs             | 50                |
+-------------------------+-------------------+
```

Call <font color=#FF0000>**history.plot_convergence()**</font> to visualize the optimization process:

```python
history.plot_convergence(true_minimum=0.397887)
```

<p align="center">
<img src="https://raw.githubusercontent.com/thomas-young-2013/open-box/master/docs/imgs/plot_convergence_branin.png" width="60%">
</p>

If you are using the Jupyter Notebook environment, call <font color=#FF0000>**history.visualize_jupyter()**</font> for visualization of 
each trial:

```python
history.visualize_jupyter()
```

<p align="center">
<img src="https://raw.githubusercontent.com/thomas-young-2013/open-box/master/docs/imgs/visualize_jupyter_branin.png" width="90%">
</p>

Call <font color=#FF0000>**print(history.get_importance())**</font> to print the hyperparameter importance
(Note that you need to install the `pyrfr` package to use this function. [Pyrfr Installation Guide](../installation/install_pyrfr.md)):

```python
print(history.get_importance())
```

```python
+------------+------------+
| Parameters | Importance |
+------------+------------+
| x1         | 0.488244   |
| x2         | 0.327570   |
+------------+------------+
```
