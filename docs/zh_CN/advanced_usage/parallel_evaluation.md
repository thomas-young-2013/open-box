# 并行评估

大多数贝叶斯优化方法（BO）只对搜索空间进行顺序的探索。
为了充分利用并行设施中的计算资源，OpenBox提出了一种分布式并行化的机制。
在这种机制中，多个配置可以在多个worker上并发地被评估。

我们考虑了两种并行机制：

<p align="center">
<img src="https://raw.githubusercontent.com/thomas-young-2013/open-box/master/docs/imgs/parallel_bo.svg" width="90%">
</p>

1) **同步并行（左）**：每个worker从推荐配置中选择一个配置进行评估。直到所有worker都完成这一轮的评估后再开始下一轮评估。

2) **异步并行（右）**：每个worker从推荐配置中选择一个配置进行评估。对于每个worker，当前的评估结束后就立刻开始下一轮评估。

**OpenBox** 提出了一种基于局部惩罚的并行化机制。其目标是对有前途的新配置进行采样，这个采样要与其它worker正在评估的配置相差足够远。
这种机制可以处理众所周知的exploration与exploitation之间的权衡，同时防止工人探索类似的配置。

在本教程中，我们将演示如何使用**OpenBox**在本地计算机上以并行的方式解决优化问题。


## 问题描述


首先，定义搜索的**配置空间**和想要<font color=#FF0000>**最小化**</font>的**目标函数**。这里我们使用**Branin**函数。


```python
import numpy as np
from openbox.utils.config_space import ConfigurationSpace, UniformFloatHyperparameter

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

如果你对这个问题描述还不熟悉，请参考我们的[快速入门教程](../quick_start/quick_start)。

## 单机并行测试

这里我们使用 <font color=#FF0000>**pSMBO**</font> 来以并行的方式在单机上优化目标函数。


```python
from openbox.optimizer.parallel_smbo import pSMBO

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

除了被传递给 **pSMBO** 的 **objective_function** 和 **config_space**，其它的参数有：

+ **parallel_strategy='async' / 'sync'** 设置并行评估是同步的还是异步的。
我们推荐使用 **'async'** 因为它能更充分地利用资源，并比 **'sync'** 实现了更好的性能。

+ **batch_size=4** 设置并行worker的数量。

+ **batch_strategy='median_imputation'** 设置如何同时提出多个建议的策略。
我们推荐使用默认参数 **'median_imputation'** 来获取稳定的性能。

+ **num_objs=1** 和 **num_constraints=0** 表明我们的函数返回一个没有限制的单目标值。

+ **max_runs=100** 表明优化过程循环100次 (优化目标函数100次). 

+ **surrogate_type='gp'** 对于数学问题，我们推荐使用高斯过程(**'gp'**)作为贝叶斯优化的替代模型。 
  对于实际问题，比如超参数优化（HPO），我们推荐使用随机森林(**'prf'**)。

+ **time_limit_per_trial** 设置每次目标函数评估的时间预算（秒）。
  一旦评估时间超过了这个限制，目标函数返回一个失败的测试。
  
+ **task_id** 指明优化过程。

在优化完成后, 调用 <font color=#FF0000>**print(bo.get_history())**</font> 来产生输出结果:

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
