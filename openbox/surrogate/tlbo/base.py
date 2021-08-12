# License: MIT

import abc
import time
import typing
import numpy as np
from typing import List

from openbox.utils.util_funcs import get_types
from openbox.core.base import build_surrogate
from openbox.utils.constants import VERY_SMALL_NUMBER
from openbox.utils.config_space import ConfigurationSpace
from openbox.utils.config_space.util import convert_configurations_to_array
from openbox.utils.normalization import zero_mean_unit_var_normalization, zero_one_normalization
from openbox.utils.logging_utils import get_logger


class BaseTLSurrogate(object):
    def __init__(self, config_space: ConfigurationSpace,
                 source_hpo_data: List,
                 seed: int,
                 history_dataset_features: List = None,
                 num_src_hpo_trial: int = 50,
                 surrogate_type='rf'):
        self.method_id = None
        self.config_space = config_space
        self.random_seed = seed
        self.num_src_hpo_trial = num_src_hpo_trial
        self.source_hpo_data = source_hpo_data
        self.source_surrogates = None
        self.target_surrogate = None
        self.history_dataset_features = history_dataset_features
        # The number of en problems.
        if source_hpo_data is not None:
            self.K = len(source_hpo_data)
            if history_dataset_features is not None:
                assert len(history_dataset_features) == self.K
        self.surrogate_type = surrogate_type

        self.types, self.bounds = get_types(config_space)
        self.instance_features = None
        self.var_threshold = VERY_SMALL_NUMBER
        self.w = None
        self.eta_list = list()

        # meta features.
        self.meta_feature_scaler = None
        self.meta_feature_imputer = None

        self.target_weight = list()
        self.logger = get_logger(self.__class__.__name__)

    @abc.abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray):
        pass

    @abc.abstractmethod
    def predict(self, X: np.ndarray):
        pass

    def build_source_surrogates(self, normalize):
        if self.source_hpo_data is None:
            self.logger.warning('No history BO data provided, resort to naive BO optimizer without TL.')
            return

        self.logger.info('Start to train base surrogates.')
        start_time = time.time()
        self.source_surrogates = list()
        for hpo_evaluation_data in self.source_hpo_data:
            print('.', end='')
            model = build_surrogate(self.surrogate_type, self.config_space,
                                    np.random.RandomState(self.random_seed))
            _X, _y = list(), list()
            for _config, _config_perf in hpo_evaluation_data.items():
                _X.append(_config)
                _y.append(_config_perf)
            X = convert_configurations_to_array(_X)
            y = np.array(_y, dtype=np.float64)
            if self.num_src_hpo_trial != -1:
                X = X[:self.num_src_hpo_trial]
                y = y[:self.num_src_hpo_trial]

            if normalize == 'standardize':
                if (y == y[0]).all():
                    y[0] += 1e-4
                y, _, _ = zero_mean_unit_var_normalization(y)
            elif normalize == 'scale':
                if (y == y[0]).all():
                    y[0] += 1e-4
                y, _, _ = zero_one_normalization(y)
                y = 2 * y - 1.
            else:
                raise ValueError('Invalid parameter in norm.')

            self.eta_list.append(np.min(y))
            model.train(X, y)
            self.source_surrogates.append(model)
        self.logger.info('Building base surrogates took %.3fs.' % (time.time() - start_time))

    def build_single_surrogate(self, X: np.ndarray, y: np.array, normalize):
        assert normalize in ['standardize', 'scale', 'none']
        model = build_surrogate(self.surrogate_type, self.config_space, np.random.RandomState(self.random_seed))
        if normalize == 'standardize':
            if (y == y[0]).all():
                y[0] += 1e-4
            y, _, _ = zero_mean_unit_var_normalization(y)
        elif normalize == 'scale':
            if (y == y[0]).all():
                y[0] += 1e-4
            y, _, _ = zero_one_normalization(y)
        else:
            pass
        model.train(X, y)
        return model

    def predict_marginalized_over_instances(self, X: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray]:
        """Predict mean and variance marginalized over all instances.

        Returns the predictive mean and variance marginalised over all
        instances for a set of configurations.

        Parameters
        ----------
        X : np.ndarray
            [n_samples, n_features (config)]

        Returns
        -------
        means : np.ndarray of shape = [n_samples, 1]
            Predictive mean
        vars : np.ndarray  of shape = [n_samples, 1]
            Predictive variance
        """

        if len(X.shape) != 2:
            raise ValueError('Expected 2d array, got %dd array!' % len(X.shape))
        if X.shape[1] != len(self.bounds):
            raise ValueError('Rows in X should have %d entries but have %d!' %
                             (len(self.bounds), X.shape[1]))

        if self.instance_features is None or \
                len(self.instance_features) == 0:
            mean, var = self.predict(X)
            assert var is not None  # please mypy

            var[var < self.var_threshold] = self.var_threshold
            var[np.isnan(var)] = self.var_threshold
            return mean, var
        raise ValueError('Unexpected case happened.')

    def combine_predictions(self, X: np.array,
                            combination_method: str = 'idp_lc',
                            weight: np.array = None):
        n, m = X.shape[0], len(self.w)
        mu, var = np.zeros((n, 1)), np.zeros((n, 1))
        if weight is None:
            w = self.w
        else:
            w = weight

        var_buf = np.zeros((n, m))
        mu_buf = np.zeros((n, m))

        target_var = None
        # Predictions from en surrogates.
        for i in range(0, self.K + 1):
            if i == self.K:
                if self.target_surrogate is not None:
                    _mu, _var = self.target_surrogate.predict(X)
                    target_var = _var
                else:
                    _mu, _var = np.zeros((n, 1)), np.zeros((n, 1))
                    raise ValueError('Target surrogate is none.')
            else:
                _mu, _var = self.source_surrogates[i].predict(X)
            mu += w[i] * _mu
            var += w[i] * w[i] * _var

            # compute the gaussian experts.
            if combination_method == 'gpoe':
                _mu, _var = _mu.flatten(), _var.flatten()
                if (_var != 0).all():
                    var_buf[:, i] = (1. / _var * w[i])
                    mu_buf[:, i] = (1. / _var * _mu * w[i])

        if combination_method == 'no_var':
            return mu, target_var
        elif combination_method == 'idp_lc':
            return mu, var
        elif combination_method == 'gpoe':
            tmp = np.sum(var_buf, axis=1)
            tmp[tmp == 0.] = 1e-5
            var = 1. / tmp
            mu = np.sum(mu_buf, axis=1) * var
            return mu.reshape(-1, 1), var.reshape(-1, 1)
        else:
            raise ValueError('Invalid combination method %s.' % combination_method)

    def scale_fit_meta_features(self, meta_features):
        from sklearn.preprocessing import MinMaxScaler, Imputer
        meta_features = np.array(meta_features)
        self.meta_feature_imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
        self.meta_feature_imputer.fit(meta_features)
        meta_features = self.meta_feature_imputer.transform(meta_features)
        self.meta_feature_scaler = MinMaxScaler()
        self.meta_feature_scaler.fit(meta_features)
        return self.meta_feature_scaler.transform(meta_features)

    def scale_transform_meta_features(self, meta_feature):
        _meta_features = np.array([meta_feature])
        _meta_feature = self.meta_feature_imputer.transform(_meta_features)
        _meta_feature = self.meta_feature_scaler.transform(_meta_feature)
        return np.clip(_meta_feature, 0, 1)
