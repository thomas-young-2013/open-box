from typing import List, Tuple
from itertools import product

import numpy as np
from scipy.stats import norm
from sklearn.kernel_approximation import RBFSampler

from litebo.acquisition_function.acquisition import AbstractAcquisitionFunction, Uncertainty
from litebo.surrogate.base.base_model import AbstractModel
from litebo.surrogate.base.gp import GaussianProcess

from platypus import NSGAII, Problem, Real


class EHVI(AbstractAcquisitionFunction):
    r"""Analytical Expected Hypervolume Improvement supporting m>=2 outcomes.

    This assumes minimization.

    Code is adapted from botorch. See [Daulton2020qehvi]_ for details.
    """

    def __init__(
        self,
        model: List[AbstractModel],
        ref_point,
        **kwargs
    ):
        """Constructor

        Parameters
        ----------
        model: A fitted model.
        ref_point: A list with `m` elements representing the reference point (in the
            outcome space) w.r.t. to which compute the hypervolume. This is a
            reference point for the objective values (i.e. after applying
            `objective` to the samples).
        """
        super().__init__(model=model, **kwargs)
        self.long_name = 'Expected Hypervolume Improvement'
        ref_point = np.asarray(ref_point)
        self.ref_point = ref_point
        self._cross_product_indices = np.array(
            list(product(*[[0, 1] for _ in range(ref_point.shape[0])]))
        )

    def psi(self, lower: np.ndarray, upper: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> None:
        r"""Compute Psi function for minimization.

        For each cell i and outcome k:

            Psi(lower_{i,k}, upper_{i,k}, mu_k, sigma_k) = (
            sigma_k * PDF((upper_{i,k} - mu_k) / sigma_k) + (
            mu_k - lower_{i,k}
            ) * (1-CDF(upper_{i,k} - mu_k) / sigma_k)

        See Equation 19 in [Yang2019]_ for more details.

        Args:
            lower: A `num_cells x m`-dim array of lower cell bounds
            upper: A `num_cells x m`-dim array of upper cell bounds
            mu: A `batch_shape x 1 x m`-dim array of means
            sigma: A `batch_shape x 1 x m`-dim array of standard deviations (clamped).

        Returns:
            A `batch_shape x num_cells x m`-dim array of values.
        """
        u = (upper - mu) / sigma
        return sigma * norm.pdf(u) + (mu - lower) * (1 - norm.cdf(u))

    def nu(self, lower: np.ndarray, upper: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> None:
        r"""Compute Nu function for minimization.

        For each cell i and outcome k:

            nu(lower_{i,k}, upper_{i,k}, mu_k, sigma_k) = (
            upper_{i,k} - lower_{i,k}
            ) * (1-CDF((upper_{i,k} - mu_k) / sigma_k))

        See Equation 25 in [Yang2019]_ for more details.

        Args:
            lower: A `num_cells x m`-dim array of lower cell bounds
            upper: A `num_cells x m`-dim array of upper cell bounds
            mu: A `batch_shape x 1 x m`-dim array of means
            sigma: A `batch_shape x 1 x m`-dim array of standard deviations (clamped).

        Returns:
            A `batch_shape x num_cells x m`-dim array of values.
        """
        return (upper - lower) * (1 - norm.cdf((upper - mu) / sigma))

    def _compute(self, X: np.ndarray, **kwargs):
        num_objs = len(self.model)
        mu = np.zeros((X.shape[0], 1, num_objs))
        sigma = np.zeros((X.shape[0], 1, num_objs))
        for i in range(num_objs):
            mean, variance = self.model[i].predict_marginalized_over_instances(X)
            sigma[:, :, i] = np.sqrt(variance)
            mu[:, :, i] = -mean

        cell_upper_bounds = np.clip(-self.cell_lower_bounds, -1e8, 1e8)

        psi_lu = self.psi(
            lower=-self.cell_upper_bounds,
            upper=cell_upper_bounds,
            mu=mu,
            sigma=sigma
        )
        psi_ll = self.psi(
            lower=-self.cell_upper_bounds,
            upper=-self.cell_upper_bounds,
            mu=mu,
            sigma=sigma
        )
        nu = self.nu(
            lower=-self.cell_upper_bounds,
            upper=cell_upper_bounds,
            mu=mu,
            sigma=sigma
        )
        psi_diff = psi_ll - psi_lu

        # This is batch_shape x num_cells x 2 x m
        stacked_factors = np.stack([psi_diff, nu], axis=-2)

        def gather(arr, index, axis):
            data_swaped = np.swapaxes(arr, 0, axis)
            index_swaped = np.swapaxes(index, 0, axis)
            gathered = np.choose(index_swaped, data_swaped)
            return np.swapaxes(gathered, 0, axis)

        # Take the cross product of psi_diff and nu across all outcomes
        # e.g. for m = 2
        # for each batch and cell, compute
        # [psi_diff_0, psi_diff_1]
        # [nu_0, psi_diff_1]
        # [psi_diff_0, nu_1]
        # [nu_0, nu_1]
        # This array has shape: `batch_shape x num_cells x 2^m x m`
        indexer = np.broadcast_to(self._cross_product_indices, stacked_factors.shape[:-2] + self._cross_product_indices.shape)
        all_factors_up_to_last = gather(stacked_factors, indexer, axis=-2)

        # Compute product for all 2^m terms, and sum across all terms and hypercells
        return all_factors_up_to_last.prod(axis=-1).sum(axis=-1).sum(axis=-1).reshape(-1, 1)


class EHVIC(EHVI):
    r"""Expected Hypervolume Improvement with Constraints, supporting m>=2 outcomes.

    This assumes minimization.

    Code is adapted from botorch. See [Daulton2020qehvi]_ for details.
    """

    def __init__(
        self,
        model: List[AbstractModel],
        constraint_models: List[GaussianProcess],
        ref_point,
        **kwargs
    ):
        """Constructor

        Parameters
        ----------
        model: A fitted model.
        ref_point: A list with `m` elements representing the reference point (in the
            outcome space) w.r.t. to which compute the hypervolume. This is a
            reference point for the objective values (i.e. after applying
            `objective` to the samples).
        """
        super().__init__(model=model, ref_point=ref_point, **kwargs)
        self.constraint_models = constraint_models
        self.long_name = 'Expected Hypervolume Improvement with Constraints'

    def _compute(self, X: np.ndarray, **kwargs):
        # Compute EHVI value
        acq = super()._compute(X)

        # Multiply by probability of feasibility
        for c_model in self.constraint_models:
            m, v = c_model.predict_marginalized_over_instances(X)
            s = np.sqrt(v)
            acq *= norm.cdf(-m / s)
        return acq


class MaxvalueEntropySearch(object):  # todo name min?
    def __init__(self, model, X, Y, beta=1e6, random_state=1):
        self.model = model  # GP model
        self.X = X
        self.Y = Y
        self.beta = beta  # todo what is beta?
        self.rbf_features = None
        self.weights_mu = None
        self.L = None
        self.sampled_weights = None
        self.random_state = random_state

    def Sampling_RFM(self):
        self.rbf_features = RBFSampler(gamma=1 / (2 * self.model.kernel.length_scale ** 2),
                                       n_components=1000, random_state=self.random_state)  # todo length_scale
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
        return X_features.dot(self.sampled_weights)

    def __call__(self, X, minimum):
        """Computes the MESMO value of single objective.

        Parameters
        ----------
        X: np.ndarray(N, D), The input points where the acquisition function
            should be evaluated. The dimensionality of X is (N, D), with N as
            the number of points to evaluate at and D is the number of
            dimensions of one X.
        minimum: float, Min value of objective (of sampled pareto front).

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

        minimum = min(minimum, min(self.Y) - 5 / self.beta)

        normalized_min = (m - minimum) / s  # todo confirm
        pdf = norm.pdf(normalized_min)
        cdf = norm.cdf(normalized_min)
        cdf[cdf == 0.0] = 1e-30
        return (normalized_min * pdf) / (2 * cdf) - np.log(cdf)


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
                 random_state=1,
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
        sample_num : int

        random_state :

        """

        super(MESMO, self).__init__(model)
        self.long_name = 'Multi-Objective Max-value Entropy Search'
        self.sample_num = sample_num
        self.random_state = random_state
        self.types = np.asarray(types)
        self.bounds = np.asarray(bounds)
        self.X = None
        self.Y = None
        self.X_dim = None
        self.Y_dim = None
        self.Multiplemes = None
        self.min_samples = None
        self.check_types_bounds()

    def check_types_bounds(self):
        # todo
        for i, (t, b) in enumerate(zip(self.types, self.bounds)):
            if np.isnan(b[1]):
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
            self.Multiplemes[i] = MaxvalueEntropySearch(self.model[i], self.X, self.Y[:, i],
                                                        random_state=self.random_state)
            self.Multiplemes[i].Sampling_RFM()

        self.min_samples = []
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
            # picking the min over the pareto: best case
            min_of_functions = [min(f) for f in list(zip(*cheap_pareto_front))]
            self.min_samples.append(min_of_functions)

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
                multi_obj_acq_sample += self.Multiplemes[i](X, self.min_samples[j][i])
            multi_obj_acq_total += multi_obj_acq_sample
        return multi_obj_acq_total / self.sample_num


class MESMOC(AbstractAcquisitionFunction):
    r"""Computes MESMOC for multi-objective optimization

    Syrine Belakaria, Aryan Deshwal, Janardhan Rao Doppa
    Max-value Entropy Search for Multi-Objective Bayesian Optimization with Constraints. 2020
    """

    def __init__(self,
                 model: List[AbstractModel],
                 constraint_models: List[AbstractModel],
                 types: List[int],
                 bounds: List[Tuple[float, float]],
                 sample_num=1,
                 random_state=1,
                 **kwargs):
        """Constructor

        Parameters
        ----------
        model : List[AbstractEPM]
            A list of surrogate that implements at least
                 - predict_marginalized_over_instances(X)
        constraint_models : List[AbstractEPM]
            A list of constraint surrogate that implements at least
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
        sample_num : int

        random_state :

        """

        super(MESMOC, self).__init__(model)
        self.long_name = 'Multi-Objective Max-value Entropy Search with Constraints'
        self.sample_num = sample_num
        self.random_state = random_state
        self.types = np.asarray(types)
        self.bounds = np.asarray(bounds)
        self.constraint_models = constraint_models
        self.num_constraints = len(constraint_models)
        self.constraint_perfs = None
        self.X = None
        self.Y = None
        self.X_dim = None
        self.Y_dim = None
        self.Multiplemes = None
        self.Multiplemes_constraints = None
        self.min_samples = None
        self.min_samples_constraints = None
        self.check_types_bounds()

    def check_types_bounds(self):
        # todo
        for i, (t, b) in enumerate(zip(self.types, self.bounds)):
            if np.isnan(b[1]):
                self.logger.error("Only int and float hyperparameters are supported in MESMOC at present!")
                raise ValueError("Only int and float hyperparameters are supported in MESMOC at present!")

    def update(self, **kwargs):
        """
        Rewrite update to support pareto front sampling.
        """
        assert 'X' in kwargs and 'Y' in kwargs
        assert 'constraint_perfs' in kwargs
        super(MESMOC, self).update(**kwargs)

        self.X_dim = self.X.shape[1]
        self.Y_dim = self.Y.shape[1]

        self.Multiplemes = [None] * self.Y_dim
        self.Multiplemes_constraints = [None] * self.num_constraints
        for i in range(self.Y_dim):
            self.Multiplemes[i] = MaxvalueEntropySearch(self.model[i], self.X, self.Y[:, i],
                                                        random_state=self.random_state)
            self.Multiplemes[i].Sampling_RFM()
        for i in range(self.num_constraints):
            # Caution dim of self.constraint_perfs!
            self.Multiplemes_constraints[i] = MaxvalueEntropySearch(self.constraint_models[i],
                                                                    self.X, self.constraint_perfs[i])
            self.Multiplemes_constraints[i].Sampling_RFM()

        self.min_samples = []
        self.min_samples_constraints = []
        for j in range(self.sample_num):
            for i in range(self.Y_dim):
                self.Multiplemes[i].weigh_sampling()
            for i in range(self.num_constraints):
                self.Multiplemes_constraints[i].weigh_sampling()

            def CMO(xi):
                xi = np.asarray(xi)
                y = [self.Multiplemes[i].f_regression(xi)[0][0] for i in range(self.Y_dim)]
                y_c = [self.Multiplemes_constraints[i].f_regression(xi)[0][0] for i in range(self.num_constraints)]
                return y, y_c

            problem = Problem(self.X_dim, self.Y_dim, self.num_constraints)
            for k in range(self.X_dim):
                problem.types[k] = Real(self.bounds[k][0], self.bounds[k][1])  # todo other types
            problem.constraints[:] = "<=0"  # todo confirm
            problem.function = CMO
            algorithm = NSGAII(problem)
            algorithm.run(1500)
            cheap_pareto_front = [list(solution.objectives) for solution in algorithm.result]
            cheap_constraints_values = [list(solution.constraints) for solution in algorithm.result]
            # picking the min over the pareto: best case
            min_of_functions = [min(f) for f in list(zip(*cheap_pareto_front))]
            min_of_constraints = [min(f) for f in list(zip(*cheap_constraints_values))]  # todo confirm
            self.min_samples.append(min_of_functions)
            self.min_samples_constraints.append(min_of_constraints)

    def _compute(self, X: np.ndarray, **kwargs):
        """Computes the MESMOC value.

        Parameters
        ----------
        X: np.ndarray(N, D), The input points where the acquisition function
            should be evaluated. The dimensionality of X is (N, D), with N as
            the number of points to evaluate at and D is the number of
            dimensions of one X.

        Returns
        -------
        np.ndarray(N,1)
            Multi-Objective Max-value Entropy Search with Constraints of X
        """
        if len(X.shape) == 1:
            X = X[:, np.newaxis]

        multi_obj_acq_total = np.zeros(shape=(X.shape[0], 1))
        for j in range(self.sample_num):
            multi_obj_acq_sample = np.zeros(shape=(X.shape[0], 1))
            for i in range(self.Y_dim):
                multi_obj_acq_sample += self.Multiplemes[i](X, self.min_samples[j][i])
            for i in range(self.num_constraints):
                # todo confirm +-
                multi_obj_acq_sample += self.Multiplemes_constraints[i](X, self.min_samples_constraints[j][i])
            multi_obj_acq_total += multi_obj_acq_sample
        acq = multi_obj_acq_total / self.sample_num

        # set unsatisfied
        constraints = []
        for i in range(self.num_constraints):
            m, _ = self.constraint_models[i].predict_marginalized_over_instances(X)
            constraints.append(m)
        constraints = np.hstack(constraints)
        unsatisfied_idx = np.where(np.any(constraints > 0, axis=1, keepdims=True))  # todo confirm
        acq[unsatisfied_idx] = -1e10
        return acq


class MESMOC2(MESMO):
    r"""Computes MESMOC2 as acquisition value.
    """

    def __init__(self,
                 model: List[AbstractModel],
                 constraint_models: List[AbstractModel],
                 types: List[int],
                 bounds: List[Tuple[float, float]],
                 sample_num=1,
                 random_state=1,
                 **kwargs):
        """Constructor

        Parameters
        ----------
        model : List[AbstractEPM]
            A list of surrogate that implements at least
                 - predict_marginalized_over_instances(X)
        constraint_models : List[AbstractEPM]
            A list of constraint surrogate that implements at least
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
        sample_num : int

        random_state :

        """
        super(MESMOC2, self).__init__(model, types, bounds, sample_num, random_state)
        self.constraint_models = constraint_models
        self.long_name = 'MESMOC2'

    def _compute(self, X: np.ndarray, **kwargs):
        """Computes the MESMOC2 value

        Parameters
        ----------
        X: np.ndarray(N, D), The input points where the acquisition function
            should be evaluated. The dimensionality of X is (N, D), with N as
            the number of points to evaluate at and D is the number of
            dimensions of one X.

        Returns
        -------
        np.ndarray(N,1)
            MESMOC2 of X
        """
        f = super()._compute(X)
        for model in self.constraint_models:
            m, v = model.predict_marginalized_over_instances(X)
            s = np.sqrt(v)
            f *= norm.cdf(-m / s)
        return f


class USeMO(AbstractAcquisitionFunction):
    r"""Computes USeMO for multi-objective optimization

    Syrine Belakaria, Aryan Deshwal, Nitthilan Kannappan Jayakodi, Janardhan Rao Doppa
    Uncertainty-Aware Search Framework for Multi-Objective Bayesian Optimization
    AAAI 2020
    """

    def __init__(self,
                 model: List[AbstractModel],
                 types: List[int],
                 bounds: List[Tuple[float, float]],
                 acq_type='ei',
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
        acq_type:
            todo
        """

        super(USeMO, self).__init__(model)
        self.long_name = 'Uncertainty-Aware Search'
        self.types = np.asarray(types)
        self.bounds = np.asarray(bounds)
        from litebo.core.base import build_acq_func
        self.single_acq = [build_acq_func(func_str=acq_type, model=m) for m in model]
        self.uncertainty_acq = [Uncertainty(model=m) for m in model]
        self.X = None
        self.Y = None
        self.X_dim = None
        self.Y_dim = None
        self.eta = None
        self.num_data = None
        self.uncertainties = None
        self.candidates = None
        self.check_types_bounds()

    def check_types_bounds(self):
        # todo
        for i, (t, b) in enumerate(zip(self.types, self.bounds)):
            if np.isnan(b[1]):
                self.logger.error("Only int and float hyperparameters are supported in USeMO at present!")
                raise ValueError("Only int and float hyperparameters are supported in USeMO at present!")

    def update(self, **kwargs):
        """
        Rewrite update
        """
        assert 'X' in kwargs and 'Y' in kwargs
        assert 'eta' in kwargs and 'num_data' in kwargs
        super(USeMO, self).update(**kwargs)

        self.X_dim = self.X.shape[1]
        self.Y_dim = self.Y.shape[1]
        assert self.Y_dim > 1

        # update single acquisition function
        for i in range(self.Y_dim):
            self.single_acq[i].update(model=self.model[i],
                                      eta=self.eta[i],
                                      num_data=self.num_data)
            self.uncertainty_acq[i].update(model=self.model[i],
                                           eta=self.eta[i],
                                           num_data=self.num_data)

        def CMO(x):
            x = np.asarray(x)
            # minimize negative acq
            return [-self.single_acq[i](x, convert=False)[0][0] for i in range(self.Y_dim)]

        problem = Problem(self.X_dim, self.Y_dim)
        for k in range(self.X_dim):
            problem.types[k] = Real(self.bounds[k][0], self.bounds[k][1])  # todo other types
        problem.function = CMO
        algorithm = NSGAII(problem)  # todo population_size
        algorithm.run(2500)
        cheap_pareto_set = [solution.variables for solution in algorithm.result]
        # cheap_pareto_set_unique = []
        # for i in range(len(cheap_pareto_set)):
        #     if not any((cheap_pareto_set[i] == x).all() for x in self.X):   # todo convert problem? no this step?
        #         cheap_pareto_set_unique.append(cheap_pareto_set[i])
        cheap_pareto_set_unique = cheap_pareto_set

        single_uncertainty = np.array([self.uncertainty_acq[i](np.asarray(cheap_pareto_set_unique), convert=False)
                                       for i in range(self.Y_dim)])  # shape=(Y_dim, N, 1)
        single_uncertainty = single_uncertainty.reshape(self.Y_dim, -1)  # shape=(Y_dim, N)
        self.uncertainties = np.prod(single_uncertainty, axis=0)  # shape=(Y_dim,) todo normalize?
        self.candidates = np.array(cheap_pareto_set_unique)

    def _compute(self, X: np.ndarray, **kwargs):
        raise NotImplementedError  # use USeMO_Optimizer
