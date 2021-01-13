from typing import List, Tuple

import numpy as np

from litebo.acquisition_function.acquisition import AbstractAcquisitionFunction
from litebo.surrogate.base.base_model import AbstractModel


class qparEGO(AbstractAcquisitionFunction):
    def __init__(self,
                 model: List[AbstractModel],
                 **kwargs):
        super().__init__(model=model, **kwargs)
        self.mc_times = kwargs.get('mc_times', 10)

    def _compute(self, X: np.ndarray, **kwargs):
        from litebo.utils.multi_objective import get_chebyshev_scalarization

        mc_samples = np.zeros(shape=(self.mc_times, len(X), len(self.model)))
        for idx in range(len(self.model)):
            mc_samples[:, :, idx] = self.model[idx].sample_functions(X, n_funcs=self.mc_times).transpose()

        samples = mc_samples.mean(axis=0)
        weights = np.random.random_sample(len(self.model))
        weights = weights / np.sum(weights)
        scalarized_obj = get_chebyshev_scalarization(weights, samples)

        # Maximize the acq function --> Minimize the objective function
        return -scalarized_obj(samples)
