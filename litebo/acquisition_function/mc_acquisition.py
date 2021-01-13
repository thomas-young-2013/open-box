from typing import List, Tuple

import numpy as np

from litebo.acquisition_function.acquisition import AbstractAcquisitionFunction
from litebo.surrogate.base.base_model import AbstractModel


class MCEI(AbstractAcquisitionFunction):
    def __init__(self,
                 model: AbstractModel,
                 par: float = 0.0,
                 **kwargs):
        super().__init__(model=model, **kwargs)
        self.long_name = 'MC-Expected Improvement'
        self.par = par
        self.eta = None
        self.mc_times = kwargs.get('mc_times', 10)

    def _compute(self, X: np.ndarray, **kwargs):
        if self.eta is None:
            raise ValueError('No current best specified. Call update('
                             'eta=<int>) to inform the acquisition function '
                             'about the current best value.')

        mc_samples = np.zeros(shape=(self.mc_times, len(X)))
        mc_samples[:, :] = self.model.sample_functions(X, n_funcs=self.mc_times).transpose()

        mc_ei = np.maximum(self.eta - mc_samples - self.par, 0)
        ei = mc_ei.mean(axis=0)
        return ei
