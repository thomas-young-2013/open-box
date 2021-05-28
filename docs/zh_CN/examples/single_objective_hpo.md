# 单目标的黑盒优化

本教程介绍如何使用**OpenBox**为ML任务调超参数。

## 数据准备

首先，给ML模型 **准备数据**。
这里我们用sklearn中的digits数据集作为实例。


```python
# prepare your data
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits

X, y = load_digits(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=1)
```

## 问题设置

第二，定义搜索的**配置空间**和想要<font color=#FF0000>**最小化**</font>的**目标函数**。
这里，我们使用 [LightGBM](https://lightgbm.readthedocs.io/en/latest/)  -- 一个微软开发的梯度提升框架，作为分类模型。


```python
from openbox.utils.config_space import ConfigurationSpace, Configuration
from openbox.utils.config_space import UniformFloatHyperparameter, \
    CategoricalHyperparameter, Constant, UniformIntegerHyperparameter
from sklearn.metrics import balanced_accuracy_score
from lightgbm import LGBMClassifier


def get_configspace():
    cs = ConfigurationSpace()
    n_estimators = UniformIntegerHyperparameter("n_estimators", 100, 1000, default_value=500, q=50)
    num_leaves = UniformIntegerHyperparameter("num_leaves", 31, 2047, default_value=128)
    max_depth = Constant('max_depth', 15)
    learning_rate = UniformFloatHyperparameter("learning_rate", 1e-3, 0.3, default_value=0.1, log=True)
    min_child_samples = UniformIntegerHyperparameter("min_child_samples", 5, 30, default_value=20)
    subsample = UniformFloatHyperparameter("subsample", 0.7, 1, default_value=1, q=0.1)
    colsample_bytree = UniformFloatHyperparameter("colsample_bytree", 0.7, 1, default_value=1, q=0.1)
    cs.add_hyperparameters([n_estimators, num_leaves, max_depth, learning_rate, min_child_samples, subsample,
                            colsample_bytree])
    return cs


def objective_function(config: Configuration):
    params = config.get_dictionary()
    params['n_jobs'] = 2
    params['random_state'] = 47

    model = LGBMClassifier(**params)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    loss = 1 - balanced_accuracy_score(y_test, y_pred)  # minimize
    return dict(objs=(loss, ))
```

下面给出了一些如何用 [ConfigSpace](https://automl.github.io/ConfigSpace/master/index.html) 定义配置空间的提示：

+ 当我们定义 **n_estimators** 时，我们设置 **q=50**，
表示超参数配置采样的间隔是50。

+ 当我们定义 **learning_rate** 时，我们设置 **log=True**，
表示超参数的值以对数方式采样。

**objective function** 的输入是一个从 **ConfigurationSpace** 中采样的 **Configuration** 实例。
你可以调用 <font color=#FF0000>**config.get_dictionary()**</font> 来把 **Configuration** 转化成一个 Python **dict**。

在这个超参数优化任务中，一旦采样出一个新的超参数配置，我们就根据输入配置重建模型。
然后，对模型进行拟合，评价模型的预测性能。
这些步骤在目标函数中执行。

评估性能后，目标函数返回一个 <font color=#FF0000>**dict (Recommended)**</font>
其中的结果包含：

+ **'objs'**：一个 **要被最小化目标值** 的 **列表/元组**。
在这个例子中，我们只有一个目标，所以这个元组只包含一个值。

+ **'constraints**'：一个含有 **限制值** 的 **列表/元组**。
如果问题没有限制，返回 **None** 或者不要把这个 key 放入字典。 非正的限制值 (**"<=0"**) 表示可行。


除了返回字典以外，对于无限制条件的单目标优化问题，我们也可以返回一个单独的值。


## 优化

在定义了配置空间和目标函数后，我们按如下方式运行优化过程：


```python
from openbox.optimizer.generic_smbo import SMBO

# Run Optimization
bo = SMBO(objective_function,
          get_configspace(),
          num_objs=1,
          num_constraints=0,
          max_runs=100,
          surrogate_type='prf',
          time_limit_per_trial=180,
          task_id='so_hpo')
history = bo.run()
```

这里我们创建一个 <font color=#FF0000>**SMBO**</font> 实例，给他传目标函数和配置空间。
其它的参数是：

+ **num_objs=1** 和 **num_constraints=0** 表示我们的函数返回一个没有限制的单独值。

+ **max_runs=100** 表示优化会进行100轮（优化目标函数100次）。

+ **surrogate_type='prf'** 对于数学问题，我们推荐用高斯过程 (**'gp'**) 做贝叶斯优化的替代模型。
对于实际问题，比如超参数优化（HPO）问题，我们推荐使用随机森林(**'prf'**)。

+ **time_limit_per_trial** 为每个目标函数评估设定最大时间预算（单位：秒）。一旦评估时间超过这个限制，目标函数返回一个失败状态。
  
+ **task_id** 用来识别优化过程。

然后，调用 <font color=#FF0000>**bo.run()**</font> 启动优化过程。


## 可视化

在优化完成后，bo.run() 会返回优化的历史过程。或者你可以调用 <font color=#FF0000>**bo.get_history()**</font> 来获得优化历史。
接下来，调用 print(history) 来查看结果：


```python
history = bo.get_history()
print(history)
```

```
+-------------------------+----------------------+
| Parameters              | Optimal Value        |
+-------------------------+----------------------+
| colsample_bytree        | 0.800000             |
| learning_rate           | 0.018402             |
| max_depth               | 15                   |
| min_child_samples       | 15                   |
| n_estimators            | 200                  |
| num_leaves              | 723                  |
| subsample               | 0.800000             |
+-------------------------+----------------------+
| Optimal Objective Value | 0.022305877305877297 |
+-------------------------+----------------------+
| Num Configs             | 100                  |
+-------------------------+----------------------+
```

调用 <font color=#FF0000>**history.plot_convergence()**</font> 来可视化优化过程：

```python
history.plot_convergence()
```

<p align="center">
<img src="https://raw.githubusercontent.com/thomas-young-2013/open-box/master/docs/imgs/plot_convergence_hpo.png" width="60%">
</p>


如果你在用 Jupyter Notebook 环境，调用 <font color=#FF0000>**history.visualize_jupyter()**</font> 来可视化每个测试：

```python
history.visualize_jupyter()
```

<p align="center">
<img src="https://raw.githubusercontent.com/thomas-young-2013/open-box/master/docs/imgs/visualize_jupyter_hpo.png" width="90%">
</p>

调用 <font color=#FF0000>**print(history.get_importance())**</font> 来输出超参数的重要性：

```python
print(history.get_importance())
```

```python
+-------------------+------------+
| Parameters        | Importance |
+-------------------+------------+
| learning_rate     | 0.293457   |
| min_child_samples | 0.101243   |
| n_estimators      | 0.076895   |
| num_leaves        | 0.069107   |
| colsample_bytree  | 0.051856   |
| subsample         | 0.010067   |
| max_depth         | 0.000000   |
+-------------------+------------+
```

在本任务中，3个最重要的超参数是 *learning_rate*，*min_child_samples*，和 *n_estimators*。
