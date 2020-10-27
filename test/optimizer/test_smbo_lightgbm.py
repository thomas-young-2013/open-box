import os
import sys
import numpy as np
import pandas as pd

from functools import partial
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    CategoricalHyperparameter, Constant, UnParametrizedHyperparameter, UniformIntegerHyperparameter
from ConfigSpace.forbidden import ForbiddenEqualsClause, \
    ForbiddenAndConjunction
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score

sys.path.append(os.getcwd())
from litebo.optimizer.smbo import SMBO
from litebo.optimizer.parallel_smbo import pSMBO


def load_data():
    data_path = 'test/optimizer/data/spambase.csv'

    label_col = -1
    header = None
    sep = ','

    na_values = ["n/a", "na", "--", "-", "?"]
    keep_default_na = True
    df = pd.read_csv(data_path, keep_default_na=keep_default_na,
                     na_values=na_values, header=header, sep=sep)

    # Drop the row with all NaNs.
    df.dropna(how='all')

    # Clean the data where the label columns have nans.
    columns_missed = df.columns[df.isnull().any()].tolist()

    label_colname = df.columns[label_col]

    if label_colname in columns_missed:
        labels = df[label_colname].values
        row_idx = [idx for idx, val in enumerate(labels) if np.isnan(val)]
        # Delete the row with NaN label.
        df.drop(df.index[row_idx], inplace=True)

    train_y = df[label_colname].values

    # Delete the label column.
    df.drop(label_colname, axis=1, inplace=True)

    train_X = df
    return train_X, train_y


def get_cs():
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


def eval_func(params, x, y):
    params = params.get_dictionary()
    model = LightGBM(**params)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=1)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    return 1 - balanced_accuracy_score(y_test, y_pred)


class LightGBM:
    def __init__(self, n_estimators, learning_rate, num_leaves, max_depth, min_child_samples,
                 subsample, colsample_bytree, random_state=None):
        self.n_estimators = int(n_estimators)
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.subsample = subsample
        self.min_child_samples = min_child_samples
        self.colsample_bytree = colsample_bytree

        self.n_jobs = 1
        self.random_state = random_state
        self.estimator = None

    def fit(self, X, y):
        from lightgbm import LGBMClassifier
        self.estimator = LGBMClassifier(num_leaves=self.num_leaves,
                                        max_depth=self.max_depth,
                                        learning_rate=self.learning_rate,
                                        n_estimators=self.n_estimators,
                                        min_child_samples=self.min_child_samples,
                                        subsample=self.subsample,
                                        colsample_bytree=self.colsample_bytree,
                                        random_state=self.random_state,
                                        n_jobs=self.n_jobs)
        self.estimator.fit(X, y)
        return self

    def predict(self, X):
        if self.estimator is None:
            raise NotImplementedError()
        return self.estimator.predict(X)


cs = get_cs()
_x, _y = load_data()
eval = partial(eval_func, x=_x, y=_y)

print('=' * 10, 'SMBO')
bo = SMBO(eval, cs, max_runs=50, time_limit_per_trial=60, logging_dir='logs')
bo.run()
inc_value = bo.get_incumbent()
print('BO', '='*30)
print(inc_value)

print('=' * 10, 'Sync Parallel SMBO')
bo = pSMBO(eval, cs, max_runs=50, time_limit_per_trial=60, logging_dir='logs', parallel_strategy='sync', batch_size=4)
bo.run()
inc_value = bo.get_incumbent()
print('BO', '='*30)
print(inc_value)

print('=' * 10, 'Sync Parallel SMBO')
bo = pSMBO(eval, cs, max_runs=50, time_limit_per_trial=60, logging_dir='logs', parallel_strategy='async', batch_size=4)
bo.run()
inc_value = bo.get_incumbent()
print('BO', '='*30)
print(inc_value)
