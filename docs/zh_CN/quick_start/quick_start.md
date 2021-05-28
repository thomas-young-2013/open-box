# 快速上手

本教程帮助你快速运行第一个 **OpenBox** 程序。

## 空间定义

首先，使用包 **ConfigSpace** 定义一个超参数配置空间。

```python
from openbox.utils.config_space import ConfigurationSpace, UniformFloatHyperparameter

# Define Configuration Space
config_space = ConfigurationSpace()
x1 = UniformFloatHyperparameter("x1", -5, 10, default_value=0)
x2 = UniformFloatHyperparameter("x2", 0, 15, default_value=0)
config_space.add_hyperparameters([x1, x2])
```

在这个例子中，我们创建了一个空的配置空间，而后向它内部添加了两个均匀分布的浮点超参数。
第一个超参数**x1**的变化范围是-5到10，第二个超参数**x2**的变化范围是0到15。

**ConfigSpace** 包也支持其它类型的超参数。
下面是定义**Integer**和**Categorical**超参数的方法：

```python
from openbox.utils.config_space import UniformIntegerHyperparameter, CategoricalHyperparameter

i = UniformIntegerHyperparameter("i", 0, 100) 
kernel = CategoricalHyperparameter("kernel", ["rbf", "poly", "sigmoid"], default_value="rbf")
```

对于 **ConfigSpace** 更高级的用法，请参考 [ConfigSpace’s documentation](https://automl.github.io/ConfigSpace/master/index.html) 。

## 定义优化目标

第二步，定义要优化的目标函数。
注意， **OpenBox** 只能 <font color=#FF0000>**最小化**</font> 目标函数。
这里我们提供了 **Branin** 函数的另一个例子。

```python
import numpy as np

# Define Objective Function
def branin(config):
    x1, x2 = config['x1'], config['x2']
    y = (x2-5.1/(4*np.pi**2)*x1**2+5/np.pi*x1-6)**2+10*(1-1/(8*np.pi))*np.cos(x1)+10
    return y
```

目标函数的输入是一个从**ConfigurationSpace**采样的配置点，输出目标值。



## 优化

在定义了配置空间和目标函数后，我们可以运行优化过程：


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

这里我们创建了一个 <font color=#FF0000>**SMBO**</font> 实例子，给他传了目标函数 **branin** 和配置空间 **config_space**。 
其余参数的含义是：

+ **num_objs=1** 和 **num_constraints=0** 表明我们的 branin 函数返回一个没有限制的单值。


+ **max_runs=50** 表示优化过程花费50轮 （优化目标函数50次）。

+ **surrogate_type='gp'**： 对于数学问题，我们推荐用高斯过程 (**'gp'**) 作为贝叶斯优化的替代模型。
对于实际的问题，例如超参数优化 (HPO)，我们推荐用随机森林 (**'prf'**)。

+ **time_limit_per_trial** 为每个目标函数评估设定最大时间预算（单位：秒）。一旦评估时间超过这个限制，目标函数返回一个失败状态。

+ **task_id** 被用来识别优化过程。

接下来，<font color=#FF0000>**bo.run()**</font> 被调用，用来重启优化过程。

## 可视化

在优化完成后， **bo.run()** 返回优化的历史信息。
可以通过调用 <font color=#FF0000>**print(history)**</font> 来看结果：

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

调用 <font color=#FF0000>**history.plot_convergence()**</font> 来可视化优化过程：

```python
history.plot_convergence(true_minimum=0.397887)
```

<p align="center">
<img src="https://raw.githubusercontent.com/thomas-young-2013/open-box/master/docs/imgs/plot_convergence_branin.png" width="60%">
</p>

如果你在用 Jupyter Notebook 环境，调用 <font color=#FF0000>**history.visualize_jupyter()**</font> 来可视化每个测试：

```python
history.visualize_jupyter()
```

<p align="center">
<img src="https://raw.githubusercontent.com/thomas-young-2013/open-box/master/docs/imgs/visualize_jupyter_branin.png" width="90%">
</p>

调用 <font color=#FF0000>**print(history.get_importance())**</font> 来输出超参数的重要性：

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
