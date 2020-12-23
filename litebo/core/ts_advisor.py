import random
import numpy as np
from copy import deepcopy

from litebo.core.generic_advisor import Advisor
from litebo.core.base import build_surrogate
from litebo.utils.samplers import SobolSampler
from litebo.utils.history_container import HistoryContainer
from litebo.config_space import Configuration
from litebo.config_space.util import convert_configurations_to_array


class TS_Advisor(Advisor):
    """
    Thompson sampling (in trust region).

    Parameters
    ----------
    batch_size : int
        Number of points in each batch.
    """

    def __init__(self, config_space,
                 task_info,
                 initial_trials=10,
                 initial_configurations=None,
                 init_strategy='sobol',
                 history_bo_data=None,
                 optimization_strategy='bo',
                 surrogate_type='gp',
                 output_dir='logs',
                 task_id=None,
                 random_state=None,
                 batch_size=4,
                 use_trust_region=False):

        super().__init__(config_space=config_space,
                         task_info=task_info,
                         initial_trials=initial_trials,
                         initial_configurations=initial_configurations,
                         init_strategy=init_strategy,
                         history_bo_data=history_bo_data,
                         optimization_strategy=optimization_strategy,
                         surrogate_type=surrogate_type,
                         output_dir=output_dir,
                         task_id=task_id,
                         random_state=random_state)

        self.setup_bo_basics()
        if self.num_constraints > 0 and not use_trust_region:
            print('Must use trust region for constrained problem. Using trust region method.')
            use_trust_region = True
        self.use_trust_region = use_trust_region
        self.dim = len(self.config_space.get_hyperparameter_names())
        self.num_candidates = min(5000, max(2000, 200 * self.dim))

        self.task_id = task_id
        self.batch_size = batch_size

        if use_trust_region:
            # Tolerances and counters
            self.fail_tol = np.ceil(np.max([4.0 / self.batch_size, self.dim / self.batch_size]))
            self.success_tol = 10

            # Trust region sizes
            self.length_min = 0.5 ** 7
            self.length_max = 1.6
            self.length_init = 0.8

            # Initialize trust region parameters
            self.fail_count = 0
            self.success_count = 0
            self.length = self.length_init
            self.center = None  # If self.center is None, use the incumbent as the center.

    def setup_bo_basics(self):
        self.surrogate_model = build_surrogate(func_str=self.surrogate_type,
                                               config_space=self.config_space,
                                               rng=self.rng,
                                               history_hpo_data=self.history_bo_data)
        if self.num_constraints > 0:
            self.constraint_models = [build_surrogate(func_str=self.constraint_surrogate_type,
                                                      config_space=self.config_space,
                                                      rng=self.rng) for _ in range(self.num_constraints)]

    def _create_candidates(self):
        """
        Return the vector representation of candidate configs.

        Candidates are sobol sequences inside the trust region.
        """

        # Pick the center as the point with the smallest function values
        # NOTE: This may not be robust to noise, in which case the posterior mean of the GP can be used instead

        if self.use_trust_region:
            # Center of the trust region
            if self.center is None:
                x_center = random.choice(self.history_container.get_incumbents())[0].get_array()
            else:
                x_center = self.center
            lb = x_center - self.length / 2.0
            ub = x_center + self.length / 2.0
        else:
            lb = np.zeros(self.dim)  # The clipping issues are handled in sampler
            ub = np.ones(self.dim)

        # Draw a sobolev sequence perturbation in [lb, ub]
        sobol = SobolSampler(self.config_space, self.num_candidates, lb, ub)
        pert = sobol.generate(return_config=False)

        # Calculate the candidate points
        if self.use_trust_region:
            # Create a perturbation mask
            prob_pert = min(20.0 / self.dim, 1.0)
            mask = np.random.rand(self.num_candidates, self.dim) <= prob_pert

            # Make sure at least one hyperparameter is perturbed
            indices = np.where(np.sum(mask, axis=1) == 0)[0]
            mask[indices, np.random.randint(0, self.dim - 1, size=len(indices))] = 1
            X_cand = x_center.copy() * np.ones((self.num_candidates, self.dim))
            X_cand[mask] = pert[mask]

        else:
            X_cand = pert

        return X_cand

    def adjust_length(self, min_objs_batch, incumbent_value):
        """
        Adjust the length of next trust region based on the evaluations of the batch recommended in the current trust region.
        """

        if min_objs_batch[0] < incumbent_value - 1e-3 * np.abs(incumbent_value):
            self.success_count += 1
            self.fail_count = 0
        else:
            self.success_count = 0
            self.fail_count += 1

        if self.success_count == self.success_tol:  # Expand trust region
            self.length = min([2.0 * self.length, self.length_max])
            self.success_count = 0
        elif self.fail_count == self.fail_tol:  # Shrink trust region
            self.length /= 2.0
            self.fail_count = 0

    def get_suggestions(self):
        if len(self.configurations) == 0:
            X = np.array([])
        else:
            failed_configs = list() if self.max_y is None else self.failed_configurations.copy()
            X = convert_configurations_to_array(self.configurations + failed_configs)

        num_failed_trial = len(self.failed_configurations)
        failed_perfs = list() if self.max_y is None else [self.max_y] * num_failed_trial
        Y = np.array(self.perfs + failed_perfs, dtype=np.float64)

        num_config_evaluated = len(self.perfs + self.failed_configurations)
        if num_config_evaluated < self.init_num:
            batch_configs_list = self.initial_configurations[num_config_evaluated:num_config_evaluated+self.batch_size]
            num_extra_config = num_config_evaluated + self.batch_size - self.init_num
            if num_extra_config > 0:
                batch_configs_list += self.sample_random_configs(num_extra_config)
            return batch_configs_list

        if self.optimization_strategy == 'random':
            return self.sample_random_configs(1)[0]
        elif self.optimization_strategy == 'bo':
            # Train surrogate model
            self.surrogate_model.train(X, Y)

            # Train constraint surrogate model
            cX = None
            if self.num_constraints > 0:
                cX = []
                for c in self.constraint_perfs:
                    failed_c = list() if num_failed_trial == 0 else [max(c)] * num_failed_trial
                    cX.append(np.array(c + failed_c, dtype=np.float64))

                for i, model in enumerate(self.constraint_models):
                    model.train(X, cX[i])

            # Create candidates
            X_cand = self._create_candidates()
            Y_cand = self.surrogate_model.sample_functions(X_cand).reshape(-1)

            if self.num_constraints > 0:
                cX_cand = np.zeros([X_cand.shape[0], self.num_constraints])
                for i, model in enumerate(self.constraint_models):
                    cX_cand[:, i] = model.sample_functions(X_cand).reshape(-1)
                violations = np.sum(cX_cand * (cX_cand > 0), axis=1)
                feasible_indices = np.where(violations == 0)[0]
                if len(feasible_indices) < self.batch_size:
                    # Select points of minimum constraint violations
                    print('Number of feasible candidates < batch size, recommending points of minimum constraint violations.')
                    batch_configs_indices = np.argsort(violations)[:self.batch_size]
                else:
                    # Select best points within feasible region
                    print('Select within feasible region.')
                    sorted_indices = np.argsort(Y_cand)
                    mask = np.isin(sorted_indices, feasible_indices)
                    batch_configs_indices = sorted_indices[mask][:self.batch_size]
            else:
                batch_configs_indices = np.argsort(Y_cand)[:self.batch_size]

            batch_configs_list = [Configuration(self.config_space, vector=X_cand[i, :]) for i in batch_configs_indices]
            return batch_configs_list

        else:
            raise ValueError('Unknown optimization strategy: %s.' % self.optimization_strategy)
