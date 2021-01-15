import numpy as np
from skopt.sampler import Sobol, Lhs

from litebo.utils.config_space import ConfigurationSpace, Configuration
from litebo.utils.util_funcs import get_types, check_random_state


class Sampler(object):
    """
    Generate samples within the specified domain (which defaults to the whole config space).

    Users should call generate() which auto-scales the samples to the domain.

    To implement new design methodologies, subclasses should implement _generate().
    """

    def __init__(self, config_space: ConfigurationSpace,
                 size, lower_bounds=None, upper_bounds=None,
                 random_state=None):
        """
        Parameters
        ----------
        config_space : ConfigurationSpace
            ConfigurationSpace to do sampling.

        size : int N
            Number of samples.

        lower_bounds : lower bounds in [0, 1] for continuous dimensions (optional)

        upper_bounds : upper bounds in [0, 1] for continuous dimensions (optional)
        """
        self.config_space = config_space

        types, bounds = get_types(config_space)
        self.search_dims = []
        for i in range(len(types)):
            if types[i] == 0 and bounds[i][1] == 1.0:  # Integer and float
                self.search_dims.append((0.0, 1.0))
            elif types[i] > 0:  # Categorical
                self.search_dims.append(list(range(types[i])))
            else:
                raise NotImplementedError()

        self.size = size
        default_lb, default_ub = zip(*bounds)
        self.lower_bounds = np.array(default_lb) if lower_bounds is None else np.clip(lower_bounds, default_lb, default_ub)
        self.upper_bounds = np.array(default_ub) if upper_bounds is None else np.clip(upper_bounds, default_lb, default_ub)

        self.rng = check_random_state(random_state)

    def set_params(self, **params):
        """
        Set the parameters of this sampler.

        Parameters
        ----------
        **params : dict
            Generator parameters.
        Returns
        -------
        self : object
            Generator instance.
        """
        if not params:
            return self
        for key, value in params.items():
            setattr(self, key, value)

        return self

    def generate(self, return_config=True):
        """
        Create samples in the domain specified during construction.

        Returns
        -------
        configs : list
            List of N sampled configurations within domain. (return_config is True)

        X : array, shape (N, D)
            Design matrix X in the specified domain. (return_config is False)
        """
        X = self._generate()
        X = self.lower_bounds + (self.upper_bounds - self.lower_bounds) * X

        if return_config:
            configs = [Configuration(self.config_space, vector=x) for x in X]
            return configs
        else:
            return X

    def _generate(self):
        """
        Create unscaled samples.

        Returns
        -------
        X : array, shape (N, D)
            Design matrix X in the config space's domain.
        """
        raise NotImplementedError()


class SobolSampler(Sampler):
    """
    Sobol sequence sampler.
    """

    def __init__(self, config_space: ConfigurationSpace,
                 size, lower_bounds=None, upper_bounds=None,
                 random_state=None):
        """
        Parameters
        ----------
        config_space : ConfigurationSpace
            ConfigurationSpace to do sampling.

        size : int N
            Number of samples.

        lower_bounds : lower bounds in [0, 1] for continuous dimensions (optional)

        upper_bounds : upper bounds in [0, 1] for continuous dimensions (optional)

        seed : int (optional)
            Seed number for sobol sequence.
        """
        super().__init__(config_space, size, lower_bounds, upper_bounds, random_state)

    def _generate(self):
        skip = self.rng.randint(int(1e6))
        try:
            from torch.quasirandom import SobolEngine
            sobol = SobolEngine(dimension=len(self.search_dims), scramble=True, seed=skip)
            X = sobol.draw(n=self.size).numpy()
        except ImportError:
            sobol = Sobol(min_skip=skip, max_skip=skip)
            X = sobol.generate(self.search_dims, self.size)
        return X


class LatinHypercubeSampler(Sampler):
    """
    Latin hypercube sampler.
    """

    def __init__(self, config_space: ConfigurationSpace,
                 size, lower_bounds=None, upper_bounds=None,
                 criterion='maximin', iterations=10000,
                 random_state=None):
        """
        Parameters
        ----------
        config_space : ConfigurationSpace
            ConfigurationSpace to do sampling.

        size : int N
            Number of samples.

        lower_bounds : lower bounds in [0, 1] for continuous dimensions (optional)

        upper_bounds : upper bounds in [0, 1] for continuous dimensions (optional)

        criterion : str or None, default='maximin'
            When set to None, the latin hypercube is not optimized

            - 'correlation' : optimized latin hypercube by minimizing the correlation
            - 'maximin' : optimized latin hypercube by maximizing the minimal pdist
            - 'ratio' : optimized latin hypercube by minimizing the ratio
              `max(pdist) / min(pdist)`

        iterations : int
            Define the number of iterations for optimizing latin hypercube.
        """
        super().__init__(config_space, size, lower_bounds, upper_bounds, random_state)
        self.criterion = criterion
        self.iterations = iterations

    def _generate(self):
        lhs = Lhs(criterion=self.criterion, iterations=self.iterations)
        X = lhs.generate(self.search_dims, self.size, random_state=self.rng)
        return X
