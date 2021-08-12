# License: MIT

import traceback
import logging
import ConfigSpace
import ConfigSpace.hyperparameters
import ConfigSpace.util
import numpy as np
import scipy.stats as sps
import statsmodels.api as sm

from openbox.utils.history_container import HistoryContainer
from openbox.core.base import Observation
from openbox.utils.config_space.util import convert_configurations_to_array


class TPE_Advisor:
    # TODO：Add warm start
    def __init__(self, config_space,
                 min_points_in_model=None,
                 top_n_percent=15,
                 num_samples=64,
                 random_fraction=1 / 3,
                 bandwidth_factor=3,
                 min_bandwidth=1e-3,
                 task_id='default_task_id',
                 output_dir='logs',
                 random_state=1):
        self.top_n_percent = top_n_percent
        self.config_space = config_space
        self.config_space.seed(random_state)
        self.bw_factor = bandwidth_factor
        self.min_bandwidth = min_bandwidth

        self.history_container = HistoryContainer(task_id, config_space=config_space)
        self.output_dir = output_dir

        self.min_points_in_model = min_points_in_model
        if min_points_in_model is None:
            self.min_points_in_model = len(self.config_space.get_hyperparameters()) + 1

        if self.min_points_in_model < len(self.config_space.get_hyperparameters()) + 1:
            self.min_points_in_model = len(self.config_space.get_hyperparameters()) + 1

        self.num_samples = num_samples
        self.random_fraction = random_fraction

        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)

        hps = self.config_space.get_hyperparameters()

        self.kde_vartypes = ""
        self.vartypes = []

        for h in hps:
            if hasattr(h, 'choices'):
                self.kde_vartypes += 'u'
                self.vartypes += [len(h.choices)]
            else:
                self.kde_vartypes += 'c'
                self.vartypes += [0]

        self.vartypes = np.array(self.vartypes, dtype=int)

        # store precomputed probs for the categorical parameters
        self.cat_probs = []

        self.good_config_rankings = dict()
        self.kde_models = dict()
        self.logger = logging.getLogger(self.__class__.__name__)

    def update_observation(self, observation: Observation):
        self.history_container.update_observation(observation)

    def get_suggestion(self, history_container=None):
        if history_container is None:
            history_container = self.history_container

        # use default as first config
        num_config_evaluated = len(history_container.configurations)
        if num_config_evaluated == 0:
            return self.config_space.get_default_configuration()

        # fit
        self.fit_kde_models(history_container)

        # If no model is available, sample random config
        if len(self.kde_models.keys()) == 0 or self.rng.rand() < self.random_fraction:
            return self.sample_random_configs(1, history_container)[0]

        best = np.inf
        best_vector = None

        try:
            l = self.kde_models['good'].pdf
            g = self.kde_models['bad'].pdf

            minimize_me = lambda x: max(1e-32, g(x)) / max(l(x), 1e-32)

            kde_good = self.kde_models['good']
            kde_bad = self.kde_models['bad']

            for i in range(self.num_samples):
                idx = self.rng.randint(0, len(kde_good.data))
                datum = kde_good.data[idx]
                vector = []

                for m, bw, t in zip(datum, kde_good.bw, self.vartypes):

                    bw = max(bw, self.min_bandwidth)
                    if t == 0:
                        bw = self.bw_factor * bw
                        try:
                            vector.append(sps.truncnorm.rvs(-m / bw, (1 - m) / bw, loc=m, scale=bw))
                        except:
                            self.logger.warning(
                                "Truncated Normal failed for:\ndatum=%s\nbandwidth=%s\nfor entry with value %s" % (
                                    datum, kde_good.bw, m))
                            self.logger.warning("data in the KDE:\n%s" % kde_good.data)
                    else:

                        if self.rng.rand() < (1 - bw):
                            vector.append(int(m))
                        else:
                            vector.append(self.rng.randint(t))
                val = minimize_me(vector)

                if not np.isfinite(val):
                    self.logger.warning('sampled vector: %s has EI value %s' % (vector, val))
                    self.logger.warning("data in the KDEs:\n%s\n%s" % (kde_good.data, kde_bad.data))
                    self.logger.warning("bandwidth of the KDEs:\n%s\n%s" % (kde_good.bw, kde_bad.bw))
                    self.logger.warning("l(x) = %s" % (l(vector)))
                    self.logger.warning("g(x) = %s" % (g(vector)))

                    # right now, this happens because a KDE does not contain all values for a categorical parameter
                    # this cannot be fixed with the statsmodels KDE, so for now, we are just going to evaluate this one
                    # if the good_kde has a finite value, i.e. there is no config with that value in the bad kde, so it shouldn't be terrible.
                    if np.isfinite(l(vector)):
                        best_vector = vector
                        break

                if val < best:
                    best = val
                    best_vector = vector

            if best_vector is None:
                self.logger.debug(
                    "Sampling based optimization with %i samples failed -> using random configuration" % self.num_samples)
                config = self.sample_random_configs(1, history_container)[0]
            else:
                self.logger.debug(
                    'best_vector: {}, {}, {}, {}'.format(best_vector, best, l(best_vector), g(best_vector)))
                for i, hp_value in enumerate(best_vector):
                    if isinstance(
                            self.config_space.get_hyperparameter(
                                self.config_space.get_hyperparameter_by_idx(i)
                            ),
                            ConfigSpace.hyperparameters.CategoricalHyperparameter
                    ):
                        best_vector[i] = int(np.rint(best_vector[i]))
                try:
                    config = ConfigSpace.Configuration(self.config_space, vector=best_vector)
                except Exception as e:
                    self.logger.warning(("=" * 50 + "\n") * 3 +
                                        "Error converting configuration:\n%s" % best_vector +
                                        "\n here is a traceback:" +
                                        traceback.format_exc())
                    raise e

        except:
            self.logger.warning(
                "Sampling based optimization with %i samples failed\n %s \nUsing random configuration" % (
                    self.num_samples, traceback.format_exc()))
            config = self.sample_random_configs(1, history_container)[0]

        return config

    def impute_conditional_data(self, array):

        return_array = np.empty_like(array)

        for i in range(array.shape[0]):
            datum = np.copy(array[i])
            nan_indices = np.argwhere(np.isnan(datum)).flatten()

            while np.any(nan_indices):
                nan_idx = nan_indices[0]
                valid_indices = np.argwhere(np.isfinite(array[:, nan_idx])).flatten()

                if len(valid_indices) > 0:
                    # pick one of them at random and overwrite all NaN values
                    row_idx = self.rng.choice(valid_indices)
                    datum[nan_indices] = array[row_idx, nan_indices]

                else:
                    # no good point in the data has this value activated, so fill it with a valid but random value
                    t = self.vartypes[nan_idx]
                    if t == 0:
                        datum[nan_idx] = self.rng.rand()
                    else:
                        datum[nan_idx] = self.rng.randint(t)

                nan_indices = np.argwhere(np.isnan(datum)).flatten()
            return_array[i, :] = datum
        return return_array

    def fit_kde_models(self, history_container):
        num_config_successful = len(history_container.successful_perfs)
        if num_config_successful <= self.min_points_in_model - 1:
            self.logger.debug("Only %i run(s) available, need more than %s -> can't build model!" % (
                num_config_successful, self.min_points_in_model + 1))
            return

        train_configs = convert_configurations_to_array(history_container.configurations)
        train_losses = history_container.get_transformed_perfs()

        n_good = max(self.min_points_in_model, (self.top_n_percent * train_configs.shape[0]) // 100)
        # n_bad = min(max(self.min_points_in_model, ((100-self.top_n_percent)*train_configs.shape[0])//100), 10)
        n_bad = max(self.min_points_in_model, ((100 - self.top_n_percent) * train_configs.shape[0]) // 100)

        # Refit KDE for the current budget
        idx = np.argsort(train_losses)

        train_data_good = self.impute_conditional_data(train_configs[idx[:n_good]])
        train_data_bad = self.impute_conditional_data(train_configs[idx[n_good:n_good + n_bad]])

        if train_data_good.shape[0] <= train_data_good.shape[1]:
            return
        if train_data_bad.shape[0] <= train_data_bad.shape[1]:
            return

        # more expensive crossvalidation method
        # bw_estimation = 'cv_ls'

        # quick rule of thumb
        bw_estimation = 'normal_reference'

        bad_kde = sm.nonparametric.KDEMultivariate(data=train_data_bad, var_type=self.kde_vartypes,
                                                   bw=bw_estimation)
        good_kde = sm.nonparametric.KDEMultivariate(data=train_data_good, var_type=self.kde_vartypes,
                                                    bw=bw_estimation)

        bad_kde.bw = np.clip(bad_kde.bw, self.min_bandwidth, None)
        good_kde.bw = np.clip(good_kde.bw, self.min_bandwidth, None)

        self.kde_models = {
            'good': good_kde,
            'bad': bad_kde
        }

        # update probs for the categorical parameters for later sampling
        self.logger.debug(
            'done building a new model based on %i/%i split\nBest loss for this budget:%f\n\n\n\n\n' % (
                n_good, n_bad, np.min(train_losses)))

    def sample_random_configs(self, num_configs=1, history_container=None):
        """
        Sample a batch of random configurations.
        Parameters
        ----------
        num_configs

        history_container

        Returns
        -------

        """
        if history_container is None:
            history_container = self.history_container

        configs = list()
        sample_cnt = 0
        max_sample_cnt = 1000
        while len(configs) < num_configs:
            config = self.config_space.sample_configuration()
            sample_cnt += 1
            if config not in (history_container.configurations + configs):
                configs.append(config)
                sample_cnt = 0
                continue
            if sample_cnt >= max_sample_cnt:
                self.logger.warning('Cannot sample non duplicate configuration after %d iterations.' % max_sample_cnt)
                configs.append(config)
                sample_cnt = 0
        return configs

