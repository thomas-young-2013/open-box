import numpy as np
from scipy.special import gamma

from litebo.utils.config_space import ConfigurationSpace, Configuration, \
    UniformFloatHyperparameter, UniformIntegerHyperparameter
from litebo.utils.util_funcs import check_random_state


class BaseTestProblem(object):
    """
    Base class for synthetic test problems.
    """

    def __init__(self, config_space: ConfigurationSpace,
                 noise_std=0,
                 optimal_value=None,
                 optimal_point=None,
                 random_state=None):
        """
        Parameters
        ----------
        config_space : Config space of the test problem.

        noise_std : Standard deviation of the observation noise.

        optimal_value : Optimal value of the test problem.
        """
        self.config_space = config_space
        self.noise_std = noise_std
        self.optimal_value = optimal_value
        self.optimal_point = optimal_point
        self.rng = check_random_state(random_state)

    @property
    def max_hv(self) -> float:
        try:
            return self._max_hv
        except AttributeError:
            raise NotImplementedError(
                f"Problem {self.__class__.__name__} does not specify maximal "
                "hypervolume."
            )

    def __call__(self, config: Configuration):
        return self.evaluate(config)

    def evaluate(self, config: Configuration):
        X = np.array(list(config.get_dictionary().values()))
        result = self._evaluate(X)
        result['objs'] = [e + self.noise_std*self.rng.randn() for e in result['objs']]
        if 'constraint' in result:
            result['constraint'] = [e + self.noise_std*self.rng.randn() for e in result['constraint']]
        return result

    def _evaluate(self, X):
        """
        Evaluate the test function.

        Returns
        -------
        result : dict
            Result of the evaluation.
            result['objs'] is the objective value or an iterable of objective values
            result['constraints'] is an iterable of constraint values
        """
        raise NotImplementedError()


class Ackley(BaseTestProblem):
    r"""Ackley test function.

    d-dimensional function:

    :math:`f(x) = -a \exp{(-b \sqrt{\sum_{i=1}^d x_i^2/d}} - \exp{(\sum_{i=1}^d cos(c x_i)/d)} + a + \exp(1)`

    f has one minimizer for its global minimum at :math:`x_{*} = (0, 0, ..., 0)` with
    :math:`f(x_{*}) = 0`.
    """

    def __init__(self, dim=2, bounds=None, constrained=False,
                 noise_std=0, random_state=None):
        self.constrained = constrained
        if bounds is None:
            if constrained:
                lb, ub = -5, 10
            else:
                lb, ub = -10, 15
        else:
            lb, ub = bounds

        params = {f'x{i}': (lb, ub, (lb + ub)/2)
                  for i in range(1, dim+1)}
        config_space = ConfigurationSpace()
        config_space.add_hyperparameters([UniformFloatHyperparameter(e, *params[e]) for e in params])
        super().__init__(config_space, noise_std, optimal_value=0, random_state=random_state)

    def _evaluate(self, X):
        a = 20
        b = 0.2
        c = 2*np.pi
        t1 = -a*np.exp(-b*np.sqrt(np.mean(X**2)))
        t2 = -np.exp(np.mean(np.cos(c*X)))
        t3 = a + np.exp(1)
        result = dict()
        result['objs'] = [t1 + t2 + t3]
        if self.constrained:
            result['constraints'] = [np.sum(X), np.sum(X**2) - 25]
        return result


class Beale(BaseTestProblem):

    def __init__(self, noise_std=0, random_state=None):
        lb, ub = -4.5, 4.5
        dim = 2
        params = {f'x{i}': (lb, ub, (lb + ub)/2)
                  for i in range(1, dim+1)}
        config_space = ConfigurationSpace()
        config_space.add_hyperparameters([UniformFloatHyperparameter(e, *params[e]) for e in params])
        super().__init__(config_space, noise_std, optimal_value=0, random_state=random_state)

    def _evaluate(self, X):
        x1, x2 = X[..., 0], X[..., 1]
        part1 = (1.5 - x1 + x1 * x2) ** 2
        part2 = (2.25 - x1 + x1 * x2 ** 2) ** 2
        part3 = (2.625 - x1 + x1 * x2 ** 3) ** 2
        result = dict()
        result['objs'] = [part1 + part2 + part3]
        return result


class Branin(BaseTestProblem):
    r"""Branin test function.

    Two-dimensional function (usually evaluated on `[-5, 10] x [0, 15]`):

        B(x) = (x_2 - b x_1^2 + c x_1 - r)^2 + 10 (1-t) cos(x_1) + 10

    Here `b`, `c`, `r` and `t` are constants where `b = 5.1 / (4 * math.pi ** 2)`
    `c = 5 / math.pi`, `r = 6`, `t = 1 / (8 * math.pi)`
    B has 3 minimizers for its global minimum at `z_1 = (-pi, 12.275)`,
    `z_2 = (pi, 2.275)`, `z_3 = (9.42478, 2.475)` with `B(z_i) = 0.397887`.
    """

    def __init__(self, noise_std=0, random_state=None):
        params = {'x1': (-5, 10, 2.5),
                  'x2': (0, 15, 7.5)}
        config_space = ConfigurationSpace()
        config_space.add_hyperparameters([UniformFloatHyperparameter(e, *params[e]) for e in params])
        super().__init__(config_space, noise_std,
                         optimal_value=0.397887,
                         optimal_point=[(-np.pi, 12.275), (np.pi, 2.275), (9.42478, 2.475)],
                         random_state=random_state)

    def _evaluate(self, X):
        t1 = (
            X[..., 1]
            - 5.1 / (4 * np.pi ** 2) * X[..., 0] ** 2
            + 5 / np.pi * X[..., 0]
            - 6
        )
        t2 = 10 * (1 - 1 / (8 * np.pi)) * np.cos(X[..., 0])
        result = dict()
        result['objs'] = [t1 ** 2 + t2 + 10]
        return result


class Bukin(BaseTestProblem):

    def __init__(self, noise_std=0, random_state=None):
        params = {'x1': (-15.0, -5.0, -10.0),
                  'x2': (-3.0, 3.0, 0)}
        config_space = ConfigurationSpace()
        config_space.add_hyperparameters([UniformFloatHyperparameter(e, *params[e]) for e in params])
        super().__init__(config_space, noise_std,
                         optimal_value=0,
                         optimal_point=[(-10.0, 1.0)],
                         random_state=random_state)

    def _evaluate(self, X):
        part1 = 100.0 * np.sqrt(np.abs(X[..., 1] - 0.01 * X[..., 0] ** 2))
        part2 = 0.01 * np.abs(X[..., 0] + 10.0)
        result = dict()
        result['objs'] = [part1 + part2]
        return result


class Rosenbrock(BaseTestProblem):
    r"""Rosenbrock synthetic test function.

    d-dimensional function (usually evaluated on `[-5, 10]^d`):

        f(x) = sum_{i=1}^{d-1} (100 (x_{i+1} - x_i^2)^2 + (x_i - 1)^2)

    f has one minimizer for its global minimum at `z_1 = (1, 1, ..., 1)` with
    `f(z_i) = 0.0`.
    """

    def __init__(self, dim=2, constrained=False, noise_std=0, random_state=None):
        self.dim = dim
        self.constrained = constrained
        params = {f'x{i}': (-5.0, 10.0, 2.5) for i in range(1, 1+self.dim)}
        config_space = ConfigurationSpace()
        config_space.add_hyperparameters([UniformFloatHyperparameter(e, *params[e]) for e in params])
        super().__init__(config_space, noise_std,
                         optimal_value=0,
                         optimal_point=[tuple(1.0 for _ in range(self.dim))],
                         random_state=random_state)

    def _evaluate(self, X):
        result = dict()
        result['objs'] = [np.sum(100.0 * (X[..., 1:] - X[..., :-1] ** 2) ** 2
                                 + (X[..., :-1] - 1) ** 2, axis=-1)]
        if self.constrained:
            result['constraints'] = [np.sum(X**2) - 2]
        return result


class Mishra(BaseTestProblem):
    r"""Mishra's Bird function (constrained).
    """

    def __init__(self, noise_std=0, random_state=None):
        params = {'x1': (-10, 0, -5), 'x2': (-6.5, 0, -3.25)}
        config_space = ConfigurationSpace()
        config_space.add_hyperparameters([UniformFloatHyperparameter(e, *params[e]) for e in params])
        super().__init__(config_space, noise_std,
                         optimal_value=-106.7645367,
                         optimal_point=[(-3.1302468, -1.5821422)],
                         random_state=random_state)

    def _evaluate(self, X):
        result = dict()
        x, y = X[0], X[1]
        t1 = np.sin(y) * np.exp((1 - np.cos(x))**2)
        t2 = np.cos(x) * np.exp((1 - np.sin(y))**2)
        t3 = (x - y)**2
        result['objs'] = [t1 + t2 + t3]
        result['constraints'] = [np.sum((X + 5)**2) - 25]
        return result


class Keane(BaseTestProblem):
    r"""Keane test function.

    d-dimensional function:

    :math:``

    f has one minimizer for its global minimum at :math:`x_{*} = (0, 0, ..., 0)` with
    :math:`f(x_{*}) = 0`.
    """

    def __init__(self, dim=2, bounds=None,
                 noise_std=0, random_state=None):
        self.dim = dim
        params = {f'x{i}': (0, 10, 5) for i in range(1, 1+self.dim)}
        config_space = ConfigurationSpace()
        config_space.add_hyperparameters([UniformFloatHyperparameter(e, *params[e]) for e in params])
        super().__init__(config_space, noise_std, optimal_value=0, random_state=random_state)

    def _evaluate(self, X):
        result = dict()
        cosX2 = np.cos(X)**2
        up = np.abs(np.sum(cosX2**2) - 2*np.prod(cosX2))
        down = np.sqrt(np.sum(np.arange(1, self.dim+1) * X**2))
        result['objs'] = [-up/down]
        result['constraints'] = [0.75 - np.prod(X), np.sum(X) - 7.5 * 30]
        return result


class Simionescu(BaseTestProblem):

    def __init__(self, noise_std=0, random_state=None):
        params = {f'x{i}': (-1.25, 1.25, 1) for i in [1, 2]}
        config_space = ConfigurationSpace()
        config_space.add_hyperparameters([UniformFloatHyperparameter(e, *params[e]) for e in params])
        super().__init__(config_space, noise_std,
                         optimal_value=-0.072,
                         optimal_point=[(0.84852813, -0.84852813), (-0.84852813, 0.84852813)],
                         random_state=random_state)

    def _evaluate(self, X):
        result = dict()
        x, y = X[0], X[1]
        result['objs'] = [0.1 * x * y]
        result['constraints'] = [x**2 + y**2 - (1 + 0.2 * np.cos(8*np.arctan(x/y)))**2]
        return result


class Rao(BaseTestProblem):
    r"""Mixed integer with constraints."""

    def __init__(self, bounds=None, noise_std=0, random_state=None):
        if bounds is None:
            bounds = [0, 20]
        lb, ub = bounds
        config_space = ConfigurationSpace()
        config_space.add_hyperparameter(UniformFloatHyperparameter('x1', lb, ub, 1))
        config_space.add_hyperparameter(UniformIntegerHyperparameter('x2', lb, ub, 1))
        super().__init__(config_space, noise_std,
                         optimal_value=-31.9998,
                         optimal_point=[(5.333, 4)],
                         random_state=random_state)

    def _evaluate(self, X):
        result = dict()
        x, y = X[0], X[1]
        result['objs'] = [-(3*x + 4*y)]
        result['constraints'] = [3*x - y - 12, 3*x + 11*y - 66]
        return result


class DTLZ(BaseTestProblem):
    r"""Base class for DTLZ problems.

    See [Deb2005dtlz]_ for more details on DTLZ.
    """

    def __init__(self, dim, num_objs=2, num_constraints=0, noise_std=0, random_state=None):
        if dim <= num_objs:
            raise ValueError(
                "dim must be > num_objs, but got %s and %s" % (dim, num_objs)
            )
        self.num_objs = num_objs
        self.num_constraints = num_constraints
        self.dim = dim
        self.k = self.dim - self.num_objs + 1
        self.bounds = [(0.0, 1.0) for _ in range(self.dim)]
        self.ref_point = [self._ref_val for _ in range(self.num_objs)]
        params = {f'x{i}': (0, 1, 0.5) for i in range(1, dim+1)}
        config_space = ConfigurationSpace()
        config_space.add_hyperparameters([UniformFloatHyperparameter(e, *params[e]) for e in params])
        super().__init__(config_space, noise_std, random_state=random_state)


class DTLZ1(DTLZ):
    r"""DLTZ1 test problem.

    d-dimensional problem evaluated on `[0, 1]^d`:

        f_0(x) = 0.5 * x_0 * (1 + g(x))
        f_1(x) = 0.5 * (1 - x_0) * (1 + g(x))
        g(x) = 100 * \sum_{i=m}^{n-1} (
        k + (x_i - 0.5)^2 - cos(20 * pi * (x_i - 0.5))
        )

    where k = n - m + 1.

    The pareto front is given by the line (or hyperplane) \sum_i f_i(x) = 0.5.
    The goal is to minimize both objectives. The reference point comes from [Yang2019]_.
    """

    _ref_val = 400.0

    def __init__(self, dim, num_objs=2, constrained=False,
                 noise_std=0, random_state=None):
        self.constrained = constrained
        super().__init__(dim, num_objs, noise_std=noise_std, random_state=random_state)

    @property
    def _max_hv(self) -> float:
        return self._ref_val ** self.num_objs - 1 / 2 ** self.num_objs

    def _evaluate(self, X):
        X_m = X[..., -self.k :]
        X_m_minus_half = X_m - 0.5
        sum_term = np.sum(X_m_minus_half**2 - np.cos(20 * np.pi * X_m_minus_half), axis=-1)
        g_X_m = 100 * (self.k + sum_term)
        g_X_m_term = 0.5 * (1 + g_X_m)
        fs = []
        for i in range(self.num_objs):
            idx = self.num_objs - 1 - i
            f_i = g_X_m_term * X[..., :idx].prod(dim=-1)
            if i > 0:
                f_i *= 1 - X[..., idx]
            fs.append(f_i)

        result = dict()
        result['objs'] = fs
        return result


class DTLZ2(DTLZ):
    r"""Unconstrained or constrained DLTZ2 test problem.

    d-dimensional problem evaluated on `[0, 1]^d`:

        f_0(x) = (1 + g(x)) * cos(x_0 * pi / 2)
        f_1(x) = (1 + g(x)) * sin(x_0 * pi / 2)
        g(x) = \sum_{i=m}^{n-1} (x_i - 0.5)^2

    The pareto front is given by the unit hypersphere \sum{i} f_i^2 = 1.
    Note: the pareto front is completely concave. The goal is to minimize
    both objectives.
    """

    _ref_val = 1.1
    _r = 0.2

    def __init__(self, dim, num_objs=2, constrained=False,
                 noise_std=0, random_state=None):
        self.constrained = constrained
        num_constraints = 1 if constrained else 0
        super().__init__(dim, num_objs, num_constraints, noise_std=noise_std, random_state=random_state)

    @property
    def _max_hv(self) -> float:
        if self.constrained:
            return 0.3996406303723544
        else:
            # hypercube - volume of hypersphere in R^n such that all coordinates are positive
            hypercube_vol = self._ref_val ** self.num_objs
            pos_hypersphere_vol = (
                np.pi ** (self.num_objs / 2)
                / gamma(self.num_objs / 2 + 1)
                / 2 ** self.num_objs
            )
            return hypercube_vol - pos_hypersphere_vol

    def _evaluate(self, X):
        X_m = X[..., -self.k :]
        g_X = np.sum((X_m - 0.5)**2, axis=-1)
        g_X_plus1 = 1 + g_X
        fs = []
        pi_over_2 = np.pi / 2
        for i in range(self.num_objs):
            idx = self.num_objs - 1 - i
            f_i = g_X_plus1.copy()
            f_i *= np.cos(X[..., :idx] * pi_over_2).prod(axis=-1)
            if i > 0:
                f_i *= np.sin(X[..., idx] * pi_over_2)
            fs.append(f_i)

        result = dict()
        result['objs'] = fs

        if self.constrained:
            f_X = np.atleast_2d(fs)
            term1 = (f_X - 1)**2
            mask = ~(np.eye(f_X.shape[-1]).astype(bool))
            indices = np.repeat(np.arange(f_X.shape[1]).reshape(1, -1), f_X.shape[1], axis=0)
            indexer = indices[mask].reshape(1, f_X.shape[1], f_X.shape[-1] - 1)
            term2_inner = np.take(np.repeat(np.expand_dims(f_X, 1), f_X.shape[-1], axis=1),
                                  indices=indexer)
            term2 = (term2_inner**2 - self._r ** 2).sum(axis=-1)
            min1 = (term1 + term2).min(axis=-1)
            min2 = ((f_X - 1 / np.sqrt(f_X.shape[-1]))**2 - self._r ** 2).sum(axis=-1)
            result['constraints'] = np.minimum(min1, min2).tolist()

        return result
