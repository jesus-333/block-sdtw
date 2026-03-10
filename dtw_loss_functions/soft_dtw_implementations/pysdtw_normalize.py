"""
Small extension for original pysdtw that implements the normalized version of the SoftDTW (i.e. the "divergence" version of the SoftDTW).

Note that the original implementation of the pysdtw is not included directly in this package, but it is listed as a dependency.

Authors
-------
Alberto Zancanaro <alberto.zancanaro@uni.lu>
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

import pysdtw
import torch

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

class pysdtw_normalized(pysdtw.SoftDTW) :
    """
    Extension of the original PySDTW implementation to include normalization (i.e. the "divergence" version of the SoftDTW).
    """

    def __init__(self, use_cuda : bool, gamma : float = 1, bandwidth : int = None, dist_func : callable = None) :
        super().__init__(use_cuda = use_cuda, gamma = gamma, bandwidth = bandwidth, dist_func = dist_func)

    def forward(self, x : torch.Tensor, y : torch.Tensor) -> torch.Tensor :
        """
        Computes the normalized SoftDTW (i.e. SoftDTW Divergence) distance between two time series.
        The final value is computed as SDTW(x, y) - SDTW(x, x) - SDTW(y, y).


        Parameters
        ----------
        x : torch.Tensor
            First input tensor of shape B x T x C
        y : torch.Tensor
            Second input tensor of shape B x T x C


        Returns
        -------
        sdtw_divergence : torch.Tensor
            Normalized SoftDTW distance between the two input tensors, of shape B.
        """
        return self.sdtw(x, y) - self.sdtw(x, x) - self.sdtw(y, y)

