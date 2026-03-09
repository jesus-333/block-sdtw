"""
SoftDTW module. Since there are various implementations online of the SoftDTW, this module was created as a wrapper to be able to easily switch between different implementations.

Currently, the following implementations are available:
- pytorch-softdtw-cuda by Mehran Maghoumi
- pysdtw by Antoine Loriette
- sdtw-cuda-torch by BGU-CS-VIL (implemented by Ron Shapira Weber)

Authors
-------
Alberto Zancanaro <alberto.zancanaro@uni.lu>

"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

import torch

import pysdtw
from .soft_dtw_implementations import soft_dtw_cuda_mag, pysdtw_normalize, soft_dtw_cuda_ron

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

class soft_dtw(torch.nn.Module) :
    """
    SoftDTW class. This class is a wrapper for the different implementations of the SoftDTW. 
    The implementation can be selected by passing the 'implementation' argument to the constructor. The available implementations are:
    - 'mag': pytorch-softdtw-cuda by Mehran Maghoumi
    - 'pysdtw': pysdtw by Antoine Loriette
    - 'ron' : sdtw-cuda-torch by BGU-CS-VIL (implemented by Ron Shapira Weber)

    Parameters
    ----------
    use_cuda : bool
        If true, this class will use the CUDA implementation of the SDTW.
    gamma : float, optional
        Value of the gamma hyperparameter for the SDTW. Default is 1.
    normalize : bool, optional
        If true, the SDTW divergence will be computed instead of the SDTW. Default is False.
    bandwidth : float, optional
        Sakoe-Chiba bandwidth for pruning. If the 'None' is given, no pruning is applied. Default is None.
    dist_func : function, optional
        Distance function to use for the SDTW. Default is None, which corresponds to the squared Euclidean distance.
    implementation : str, optional
        Implementation to use for the SDTW.
    fused : bool, optional
        Only for the 'ron' implementation.
        None  -> auto (use fused only when possible)
        True  -> require fused (error if not possible)
        False -> never fused (always materialize D and use D-based autograd)
    """

    def __init__(self, use_cuda : bool,
                 gamma : float = 1, normalize : bool = False, bandwith : int = None,
                 dist_func : callable = None,
                 implementation : str = 'mag',
                 fused : bool = None) :
        super().__init__()
        
        if implementation == 'mag' :
            self.sdtw_function = soft_dtw_cuda_mag.SoftDTW(use_cuda = use_cuda, gamma = gamma, normalize = normalize, bandwidth = bandwith, dist_func = dist_func)
        elif implementation == 'pysdtw' :
            if normalize :
                self.sdtw_function = pysdtw_normalize.pysdtw_normalized(use_cuda = use_cuda, gamma = gamma, bandwidth = bandwith, dist_func = dist_func)
            else :
                self.sdtw_function = pysdtw.SoftDTW(use_cuda = use_cuda, gamma = gamma, bandwidth = bandwith, dist_func = dist_func)
        elif implementation == 'ron' :
            self.sdtw_function = soft_dtw_cuda_ron.SoftDTW(gamma = gamma, normalize = normalize, bandwidth = bandwith, dist_func = dist_func, fused = fused)

    def forward(self, x : torch.Tensor, y : torch.Tensor) -> torch.Tensor :
        """
        Compute the SoftDTW distance between two time series.

        Parameters
        ----------
        x : torch.Tensor
            First input tensor of shape B x T x C
        y : torch.Tensor
            Second input tensor of shape B x T x C

        Returns
        -------
        torch.Tensor
            SoftDTW distance between the two time series
        """

        return self.sdtw_function(x, y)

    def check_implementation(self, implementation : str) :
        """
        Check if the selected implementation is valid. If not, raise an error.
        """

        implementations = ['mag', 'pysdtw', 'ron']

        if implementation not in implementations :
            raise ValueError(f"Invalid implementation selected. Implementations available: {implementations}. Selected implementation: {implementation}.")
