import numpy as np
from litebo.surrogate.tlbo.base import BaseTLSurrogate
from litebo.core.base import build_surrogate
from litebo.config_space.util import convert_configurations_to_array


class SGPR(BaseTLSurrogate):
    def __init__(self, config_space, source_hpo_data, target_hp_configs, seed,
                 surrogate_type='rf', num_src_hpo_trial=50):
        super().__init__(config_space, source_hpo_data, seed, target_hp_configs,
                         surrogate_type=surrogate_type, num_src_hpo_trial=num_src_hpo_trial)
        self.method_id = 'sgpr'

        self.alpha = 0.95
        self.configs_set = None
        self.cached_prior_mu = None
        self.cached_prior_sigma = None
        self.cached_stacking_mu = None
        self.cached_stacking_sigma = None
        self.prior_size = 0
        self.iteration_id = 0
        self.index_mapper = dict()
        self.get_regressor()

    def get_regressor(self):
        # Collect the configs set.
        configs_list = list()

        for idx, hpo_evaluation_data in enumerate(self.source_hpo_data):
            for _config, _ in list(hpo_evaluation_data.items())[:self.num_src_hpo_trial]:
                if _config not in configs_list:
                    configs_list.append(_config)
        for _config in self.target_hp_configs:
            if _config not in configs_list:
                configs_list.append(_config)
        configs_list = [list(item) for item in convert_configurations_to_array(configs_list)]

        # Initialize mu and sigma vector.
        num_configs = len(configs_list)
        self.configs_set = configs_list
        for _idx, _config in enumerate(configs_list):
            self.index_mapper[str(_config)] = _idx

        self.cached_prior_mu = np.zeros(num_configs)
        self.cached_prior_sigma = np.ones(num_configs)
        self.cached_stacking_mu = np.zeros(num_configs)
        self.cached_stacking_sigma = np.ones(num_configs)
        self.configs_X = np.array(self.configs_set)

        # Train transfer learning regressor.
        self.prior_size = 0
        for idx, hpo_evaluation_data in enumerate(self.source_hpo_data):
            print('Build the %d-th residual GPs.' % idx)
            _X, _y = list(), list()
            for _config, _config_perf in list(hpo_evaluation_data.items())[:self.num_src_hpo_trial]:
                _X.append(_config)
                _y.append(_config_perf)
            X = convert_configurations_to_array(_X)
            y = np.array(_y, dtype=np.float64)
            self.train_regressor(X, y)
            self.prior_size = len(y)

    def train_regressor(self, X, y, is_top=False):
        model = build_surrogate(self.surrogate_type, self.config_space,
                                np.random.RandomState(self.random_seed))
        model.train(X, y)

        # Get prior mu and sigma for configs in X.
        idxs = list()
        for item in X:
            # index = self.configs_set.index(list(item))
            index = self.index_mapper[str(list(item))]
            idxs.append(index)
        prior_mu, prior_sigma = self.cached_prior_mu[idxs], self.cached_prior_sigma[idxs]
        prior_mu = np.array(prior_mu)

        # Training residual GP.
        model.train(X, y - prior_mu)

        # Update the prior surrogate: mu and sigma.
        top_size = len(y)
        beta = self.alpha * top_size / (self.alpha * top_size + self.prior_size)

        mu_top, sigma_top = model.predict(self.configs_X)
        mu_top, sigma_top = mu_top.flatten(), sigma_top.flatten()

        if is_top:
            self.cached_stacking_mu = self.cached_prior_mu + mu_top
            self.cached_stacking_sigma = np.power(sigma_top, beta) * \
                                         np.power(self.cached_prior_sigma, 1 - beta)
        else:
            self.cached_prior_mu += mu_top.flatten()
            self.cached_prior_sigma = np.power(sigma_top, beta) * \
                                      np.power(self.cached_prior_sigma, 1 - beta)

    def train(self, X: np.ndarray, y: np.array):
        # Decide whether to rebuild the transfer learning regressor.
        retrain = False
        for item in X:
            item = list(item)
            if item not in self.configs_set:
                retrain = True
                break
        if retrain:
            self.get_regressor()

        # Train the final regressor.
        self.train_regressor(X, y, is_top=True)
        self.iteration_id += 1

    def predict(self, X: np.array):
        index_list = list()
        for x in X:
            # index = self.configs_set.index(list(x))
            index = self.index_mapper[str(list(x))]
            index_list.append(index)

        mu_list, var_list = self.cached_stacking_mu[index_list], self.cached_stacking_sigma[index_list]
        return np.array(mu_list).reshape(-1, 1), np.array(var_list).reshape(-1, 1)
