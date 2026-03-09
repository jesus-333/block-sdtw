"""
Implementation of the block DTW, which is a variant of the SDTW that computes the SDTW on blocks of the signal instead of the entire signal.

Example
-------
>>> from dtw_loss_functions import block_dtw
>>> import torch
>>> block_size = 25
>>> use_cuda = torch.cuda.is_available()
>>> device = 'cuda' if use_cuda else 'cpu'
>>> batch_size = 5
>>> time_samples = 300
>>> channels = 1
>>> x   = torch.randn(batch_size, time_samples, channels).to(device)
>>> x_r = torch.randn(batch_size, time_samples, channels).to(device)
>>> block_dtw_loss = block_dtw.block_dtw(block_size, use_cuda)
>>> output_block_dtw = block_dtw_loss(x, x_r)

Authors
-------
Alberto Zancanaro <alberto.zancanaro@uni.lu>

"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

import torch

from .soft_dtw import soft_dtw

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

class block_dtw(torch.nn.Module) :
    """
    Class that compute the block DTW loss, which is a variant of the SDTW that computes the SDTW on blocks of the signal instead of the entire signal.
    The block DTW can be computed in two ways: 

    - Naive (SEQUENTIAL) implementation: compute the SDTW on each block separately and sum the results.
    - Optimized (PARALLEL) implementation: exploit reshaping of the input tensors to compute the SDTW on all blocks at once.

    This class will select automatically which implementation to use based on the input tensors length and block size.
    See the docstring of block_dtw_optimized for more details on the requirements for the optimized implementation.


    If you are not sure which implementation to use, you can use the block_dtw class.
    Note that if you know a priori that the optimized version can be used in your case, it is recommended to use directly the block_dtw_optimized class, which is faster than the block_dtw class (no overhead of checking the input tensors length and block size).

    Attributes
    ----------

    block_size : int
        Size of the blocks into which to divide the signal.
    block_dtw_naive : :class:`block_dtw_naive`
        Instance of the naive implementation of the block DTW.
    block_dtw_optimized : :class:`block_dtw_optimized`
        Instance of the optimized implementation of the block DTW.

    Parameters
    ----------
    block_size : int
        Size of the blocks into which to divide the signal.
    use_cuda : bool
        If true, this class will use the CUDA implementation of the SDTW.
    gamma_sdtw : float, optional
        Value of the gamma hyperparameter for the SDTW. Default is 1.
    use_divergence : bool, optional
        If true, compute the SDTW divergence instead of the SDTW. Default is False.
    bandwidth : float, optional
        Sakoe-Chiba bandwidth for pruning. If the 'None' is given, no pruning is applied. Default is None.
    dist_func : function, optional
        Distance function to use for the SDTW. Default is None, which corresponds to the squared Euclidean distance.
    """

    def __init__(self, block_size : int,
                 use_cuda : bool,
                 gamma_sdtw : float = 1, use_divergence : bool = False, bandwidth : float = None, dist_func = None,
                 implementation : str = 'mag',
                 fused : bool = None,
                 ) :
        super().__init__()

        self.block_size = block_size

        self.block_dtw_naive = block_dtw_naive(block_size = block_size, use_cuda = use_cuda, gamma_sdtw = gamma_sdtw, use_divergence = use_divergence, bandwidth = bandwidth, dist_func = dist_func)
        self.block_dtw_optimized = block_dtw_optimized(block_size = block_size, use_cuda = use_cuda, gamma_sdtw = gamma_sdtw, use_divergence = use_divergence, bandwidth = bandwidth, dist_func = dist_func)

    def forward(self, x : torch.tensor, x_r : torch.tensor) -> torch.tensor :
        """
        Compute the block DTW loss between the input tensors ``x`` and ``x_r``.

        Parameters
        ----------
        x : torch.tensor
            First input tensor of shape ``B x T x C``
        x_r : torch.tensor
            Second input tensor of shape ``B x T x C``

        Returns
        -------
        recon_error : torch.tensor
            Tensor of shape ``B`` containing the block DTW loss for each sample in the batch.
        """

        if x.shape[1] % self.block_size == 0 :
            return self.block_dtw_optimized(x, x_r)
        else :
            return self.block_dtw_naive(x, x_r)


class block_dtw_naive(soft_dtw) :
    """
    Naive implementation of the block DTW, which computes the SDTW on each block separately.

    For details on the parameters, see the docstring of the :class:`block_dtw` class.
    """

    def __init__(self, block_size : int,
                 use_cuda : bool,
                 gamma_sdtw : float = 1, use_divergence : bool = False, bandwidth : float = None, dist_func = None,
                 implementation : str = 'mag',
                 fused : bool = None,
                 ) :

        super().__init__(use_cuda = use_cuda, gamma = gamma_sdtw, normalize = use_divergence, bandwidth = bandwidth, dist_func = dist_func, implementation = implementation, fused = fused)

        self.block_size = block_size

    def forward(self, x : torch.tensor, x_r : torch.tensor) -> torch.tensor :
        """
        Compute the block DTW loss between the input tensors ``x`` and ``x_r`` by computing the SDTW on each block separately and summing the results.

        Parameters
        ----------
        x : torch.tensor
            First input tensor of shape ``B x T x C``
        x_r : torch.tensor
            Second input tensor of shape ``B x T x C``

        Returns
        -------
        recon_error : torch.tensor
            Tensor of shape ``B`` containing the block DTW loss for each sample in the batch.
        """
        tmp_recon_loss = 0
        i = 0
        continue_cylce = True

        while continue_cylce :
            # Get indices for the block
            idx_1 = int(i * self.block_size)
            idx_2 = int((i + 1) * self.block_size) if int((i + 1) * self.block_size) < x.shape[1] else -1

            # Get block of the signal
            if idx_2 == -1 :
                x_block = x[:, idx_1:, :]
                x_r_block = x_r[:, idx_1:, :]
            else :
                x_block = x[:, idx_1:idx_2, :]
                x_r_block = x_r[:, idx_1:idx_2, :]

            # Compute dtw for the block
            block_loss = super().forward(x_block, x_r_block)

            # End the cylce at the last block
            if idx_2 == -1 : continue_cylce = False

            tmp_recon_loss += block_loss

            # Increase index
            i += 1

        return tmp_recon_loss

class block_dtw_optimized(soft_dtw) :
    """
    Optimized implementation of the block DTW, which exploits reshaping of the input tensors to compute the SDTW on all blocks at once.
    

    This version can be used only if the length of the input tensors is divisible by the block size, i.e. if ``length_signal % block_size == 0``.
    Note that the class will not check if this condition is satisfied, so it is the responsibility of the user to ensure that the input tensors length and block size are compatible.

    This requirement is necessary because the optimized implementation exploits reshaping of the input tensors to compute the SDTW on all blocks at once.
    This works because SoftDTW implementation allow batched inputs,so we can reshape the input tensors to have a new batch size equal to the number of blocks, and compute the SDTW on all blocks at once.

    For details on the parameters, see the docstring of the :class:`block_dtw` class.
    """

    def __init__(self, block_size : int,
                 use_cuda : bool,
                 gamma_sdtw : float = 1, use_divergence : bool = False, bandwidth : float = None, dist_func = None,
                 implementation : str = 'mag',
                 fused : bool = None,
                 ) :

        super().__init__(use_cuda = use_cuda, gamma = gamma_sdtw, normalize = use_divergence, bandwidth = bandwidth, dist_func = dist_func, implementation = implementation, fused = fused)

        self.block_size = block_size

    def forward(self, x : torch.tensor, x_r : torch.tensor) -> torch.tensor :
        """
        Compute the block DTW loss between the input tensors `x` and `x_r` by exploiting reshaping of the input tensors to compute the SDTW on all blocks at once.

        Parameters
        ----------
        x : torch.tensor
            First input tensor of shape B x T x C
        x_r : torch.tensor
            Second input tensor of shape B x T x C

        Returns
        -------
        recon_error : torch.tensor
            Tensor of shape B containing the block DTW loss for each sample in the batch.
        """

        # Compute new batch size
        virtual_batch_size = int(x.shape[0] * (x.shape[1] / self.block_size))
        
        # Reshape input tensors
        x_reshaped   = x.view(virtual_batch_size, self.block_size, -1)
        x_r_reshaped = x_r.view(virtual_batch_size, self.block_size, -1)

        block_loss = super().forward(x_reshaped, x_r_reshaped)

        # Return the loss reshaped to have the original batch size
        return block_loss.view(x.shape[0], -1).sum(dim = 1)
