from typing import List, Tuple

import numpy as np
from scipy.stats import norm
from sklearn.kernel_approximation import RBFSampler

from litebo.acquisition_function.acquisition import AbstractAcquisitionFunction
from litebo.surrogate.base.base_model import AbstractModel

from platypus import NSGAII, Problem, Real  # todo requirements


class MaxvalueEntropySearch(object):
    def __init__(self, model, X, Y, beta=1e6):
        self.model = model      # GP model
        self.X = X
        self.Y = Y
        self.beta = beta     # todo what is GPmodel.beta?
        self.rbf_features = None
        self.weights_mu = None
        self.L = None
        self.sampled_weights = None

    def Sampling_RFM(self):
        self.rbf_features = RBFSampler(gamma=1 / (2 * self.model.kernel.length_scale ** 2),
                                       n_components=1000, random_state=1)       # todo random_state, length_scale
        X_train_features = self.rbf_features.fit_transform(np.asarray(self.X))

        A_inv = np.linalg.inv(
            (X_train_features.T).dot(X_train_features) + np.eye(self.rbf_features.n_components) / self.beta)
        self.weights_mu = A_inv.dot(X_train_features.T).dot(self.Y)
        weights_gamma = A_inv / self.beta
        self.L = np.linalg.cholesky(weights_gamma)

    def weigh_sampling(self):
        random_normal_sample = np.random.normal(0, 1, np.size(self.weights_mu))
        self.sampled_weights = np.c_[self.weights_mu] + self.L.dot(np.c_[random_normal_sample])

    def f_regression(self, x):
        X_features = self.rbf_features.fit_transform(x.reshape(1, len(x)))
        return -(X_features.dot(self.sampled_weights))

    def __call__(self, X, maximum):
        """Computes the MESMO value of single objective.

        Parameters
        ----------
        X: np.ndarray(N, D), The input points where the acquisition function
            should be evaluated. The dimensionality of X is (N, D), with N as
            the number of points to evaluate at and D is the number of
            dimensions of one X.
        maximum: float
            # todo
        Returns
        -------
        np.ndarray(N,1)
            Max-value Entropy Search of X
        """
        m, v = self.model.predict_marginalized_over_instances(X)
        s = np.sqrt(v)

        if np.any(s == 0.0):
            y_std = np.std(self.Y)
            if y_std == 0:
                y_std = 1
            s[s == 0.0] = np.sqrt(1e-5) * y_std

        maximum = max(maximum, max(self.Y) + 5 / self.beta)

        normalized_max = (maximum - m) / s
        pdf = norm.pdf(normalized_max)
        cdf = norm.cdf(normalized_max)
        cdf[cdf == 0.0] = 1e-30
        return -(normalized_max * pdf) / (2 * cdf) + np.log(cdf)


class MESMO(AbstractAcquisitionFunction):

    r"""Computes MESMO for multi-objective optimization

    Syrine Belakaria, Aryan Deshwal, Janardhan Rao Doppa
    Max-value Entropy Search for Multi-Objective Bayesian Optimization. NeurIPS 2019
    https://papers.nips.cc/paper/8997-max-value-entropy-search-for-multi-objective-bayesian-optimization.pdf
    """

    def __init__(self,
                 model: List[AbstractModel],
                 types: List[int],
                 bounds: List[Tuple[float, float]],
                 sample_num=1,
                 **kwargs):
        """Constructor

        Parameters
        ----------
        model : List[AbstractEPM]
            A list of surrogate that implements at least
                 - predict_marginalized_over_instances(X)
        types : List[int]
            Specifies the number of categorical values of an input dimension where
            the i-th entry corresponds to the i-th input dimension. Let's say we
            have 2 dimension where the first dimension consists of 3 different
            categorical choices and the second dimension is continuous than we
            have to pass [3, 0]. Note that we count starting from 0.
        bounds : List[Tuple[float, float]]
            Bounds of input dimensions: (lower, upper) for continuous dims; (n_cat, np.nan) for categorical dims
            # todo cat dims
        """

        super(MESMO, self).__init__(model)
        self.long_name = 'Multi-Objective Max-value Entropy Search'
        self.sample_num = sample_num
        self.types = np.asarray(types)
        self.bounds = np.asarray(bounds)
        self.X = None
        self.Y = None
        self.X_dim = None
        self.Y_dim = None
        self.Multiplemes = None
        self.max_samples = None
        self.check_types_bounds()

    def check_types_bounds(self):
        # todo
        for i, (t, b) in enumerate(zip(self.types, self.bounds)):
            if b[1] is np.nan:
                self.logger.error("Only int and float hyperparameters are supported in MESMO at present!")
                raise ValueError("Only int and float hyperparameters are supported in MESMO at present!")

    def update(self, **kwargs):
        """
        Rewrite update to support pareto front sampling.
        """
        assert 'X' in kwargs and 'Y' in kwargs
        super(MESMO, self).update(**kwargs)

        self.X_dim = self.X.shape[1]
        self.Y_dim = self.Y.shape[1]

        self.Multiplemes = [None] * self.Y_dim
        for i in range(self.Y_dim):
            self.Multiplemes[i] = MaxvalueEntropySearch(self.model[i], self.X, self.Y[:, i])
            self.Multiplemes[i].Sampling_RFM()

        self.max_samples = []
        for j in range(self.sample_num):
            for i in range(self.Y_dim):
                self.Multiplemes[i].weigh_sampling()

            def CMO(xi):
                xi = np.asarray(xi)
                y = [self.Multiplemes[i].f_regression(xi)[0][0] for i in range(self.Y_dim)]
                return y

            problem = Problem(self.X_dim, self.Y_dim)
            for k in range(self.X_dim):
                problem.types[k] = Real(self.bounds[k][0], self.bounds[k][1])  # todo other types
            problem.function = CMO
            algorithm = NSGAII(problem)
            algorithm.run(1500)
            cheap_pareto_front = [list(solution.objectives) for solution in algorithm.result]
            # picking the max over the pareto: best case
            max_of_functions = [-1 * min(f) for f in list(zip(*cheap_pareto_front))]
            self.max_samples.append(max_of_functions)

    def _compute(self, X: np.ndarray, **kwargs):
        """Computes the MESMO value.

        Parameters
        ----------
        X: np.ndarray(N, D), The input points where the acquisition function
            should be evaluated. The dimensionality of X is (N, D), with N as
            the number of points to evaluate at and D is the number of
            dimensions of one X.

        Returns
        -------
        np.ndarray(N,1)
            Multi-Objective Max-value Entropy Search of X
        """
        if len(X.shape) == 1:
            X = X[:, np.newaxis]

        multi_obj_acq_total = np.zeros(shape=(X.shape[0], 1))
        for j in range(self.sample_num):
            multi_obj_acq_sample = np.zeros(shape=(X.shape[0], 1))
            for i in range(self.Y_dim):
                multi_obj_acq_sample += self.Multiplemes[i](X, self.max_samples[j][i])
            multi_obj_acq_total += multi_obj_acq_sample
        return multi_obj_acq_total / self.sample_num
