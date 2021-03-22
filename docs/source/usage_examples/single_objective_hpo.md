# Single Objective Hyperparameter Optimization

```python
# prepare your data first
x, y = load_data(data_path)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=1)
```

```python
from litebo.utils.config_space import ConfigurationSpace
from litebo.utils.config_space import UniformFloatHyperparameter, \
    CategoricalHyperparameter, Constant, UniformIntegerHyperparameter
from sklearn.metrics import balanced_accuracy_score
from lightgbm import LGBMClassifier
from functools import partial

class LightGBM:
    def __init__(self, n_estimators, num_leaves, max_depth, learning_rate, min_child_samples,
                 subsample, colsample_bytree, random_state=None):
        self.n_estimators = int(n_estimators)
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.min_child_samples = min_child_samples
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree

        self.random_state = random_state
        self.estimator = None

    def fit(self, X, y):
        self.estimator = LGBMClassifier(n_estimators=self.n_estimators,
                                        num_leaves=self.num_leaves,
                                        max_depth=self.max_depth,
                                        learning_rate=self.learning_rate,
                                        min_child_samples=self.min_child_samples,
                                        subsample=self.subsample,
                                        colsample_bytree=self.colsample_bytree,
                                        random_state=self.random_state,
                                        n_jobs=2)
        self.estimator.fit(X, y)
        return self

    def predict(self, X):
        if self.estimator is None:
            raise NotImplementedError()
        return self.estimator.predict(X)

def get_configspace():
    cs = ConfigurationSpace()
    n_estimators = UniformFloatHyperparameter("n_estimators", 100, 1000, default_value=500, q=50)
    num_leaves = UniformIntegerHyperparameter("num_leaves", 31, 2047, default_value=128)
    max_depth = Constant('max_depth', 15)
    learning_rate = UniformFloatHyperparameter("learning_rate", 1e-3, 0.3, default_value=0.1, log=True)
    min_child_samples = UniformIntegerHyperparameter("min_child_samples", 5, 30, default_value=20)
    subsample = UniformFloatHyperparameter("subsample", 0.7, 1, default_value=1, q=0.1)
    colsample_bytree = UniformFloatHyperparameter("colsample_bytree", 0.7, 1, default_value=1, q=0.1)
    cs.add_hyperparameters([n_estimators, num_leaves, max_depth, learning_rate, min_child_samples, subsample,
                            colsample_bytree])
    return cs

def objective_function(config, x_train, x_test, y_train, y_test):
    params = config.get_dictionary()
    model = LightGBM(**params)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    loss = 1 - balanced_accuracy_score(y_test, y_pred)  # minimize
    return dict(objs=(loss, ))

objective_function = partial(objective_function, x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)
config_space = get_configspace()
```

```python
from litebo.optimizer.generic_smbo import SMBO

# Run Optimization
bo = SMBO(objective_function,
          config_space,
          num_objs=1,
          num_constraints=0,
          max_runs=100,
          surrogate_type='prf',
          time_limit_per_trial=180,
          task_id='so_hpo')
bo.run()
print(bo.get_history())
```
