"""
Implementation of the block DTW, which is a variant of the SDTW that computes the SDTW on blocks of the signal instead of the entire signal.

@author: Alberto Zancanaro (Jesus)
@organization: Luxembourg Centre for Systems Biomedicine (LCSB)
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

import torch

from .soft_dtw_cuda import SoftDTW

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

class block_dtw() :

    def __init__(self, block_size : int,
                 use_cuda : bool,
                 gamma_sdtw : float = 1, use_divergence : bool = False, bandwidth : float = None, dist_func = None
                 ) :
        """
        Class that compute the block DTW loss, which is a variant of the SDTW that computes the SDTW on blocks of the signal instead of the entire signal.
        The block DTW can be computed in two ways: 
        - Naive (SEQUENTIAL) implementation: compute the SDTW on each block separately and sum the results.
        - Optimized (PARALLEL) implementation: exploit reshaping of the input tensors to compute the SDTW on all blocks at once.
        This class will select automatically which implementation to use based on the input tensors length and block size.
        See the docstring of block_dtw_optimized for more details on the requirements for the optimized implementation.

        If you are not sure which implementation to use, you can use the block_dtw class.
        Note that if you know a priori that the optimized version can be used in your case, it is recommended to use directly the block_dtw_optimized class, which is faster than the block_dtw class (no overhead of checking the input tensors length and block size).

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
        super().__init__()

        self.block_size = block_size

        self.block_dtw_naive = block_dtw_naive(block_size = block_size, use_cuda = use_cuda, gamma_sdtw = gamma_sdtw, use_divergence = use_divergence, bandwidth = bandwidth, dist_func = dist_func)
        self.block_dtw_optimized = block_dtw_optimized(block_size = block_size, use_cuda = use_cuda, gamma_sdtw = gamma_sdtw, use_divergence = use_divergence, bandwidth = bandwidth, dist_func = dist_func)

    def __call__(self, x : torch.tensor, x_r : torch.tensor) -> torch.tensor :
        """
        Compute the block DTW loss between the input tensors x and x_r.

        Parameters
        ----------
        x : torch.tensor
            First input tensor of shape B x T x 1
        x_r : torch.tensor
            Second input tensor of shape B x T x 1

        Returns
        -------
        recon_error : torch.tensor
            Tensor of shape B containing the block DTW loss for each sample in the batch.
        """

        if x.shape[1] % self.block_size == 0 :
            return self.block_dtw_optimized(x, x_r)
        else :
            return self.block_dtw_naive(x, x_r)


class block_dtw_naive(SoftDTW) :

    def __init__(self, block_size : int,
                 use_cuda : bool,
                 gamma_sdtw : float = 1, use_divergence : bool = False, bandwidth : float = None, dist_func = None
                 ) :
        """
        Naive implementation of the block DTW, which computes the SDTW on each block separately.
        For details on the parameters, see the docstring of the block_dtw class.
        """

        super().__init__(use_cuda = use_cuda, gamma = gamma_sdtw, normalize = use_divergence, bandwidth = bandwidth, dist_func = dist_func)

        self.block_size = block_size

    def forward(self, x : torch.tensor, x_r : torch.tensor) -> torch.tensor :
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

class block_dtw_optimized(SoftDTW) :

    def __init__(self, block_size : int,
                 use_cuda : bool,
                 gamma_sdtw : float = 1, use_divergence : bool = False, bandwidth : float = None, dist_func = None
                 ) :
        """
        Optimized implementation of the block DTW, which exploits reshaping of the input tensors to compute the SDTW on all blocks at once.
        
        The optimized implementation can be used only if the length of the input tensors is divisible by the block size, i.e. if length_signal % block_size == 0.
        Note that the class will not check if this condition is satisfied, so it is the responsibility of the user to ensure that the input tensors length and block size are compatible.

        For details on the parameters, see the docstring of the block_dtw class.
        """

        super().__init__(use_cuda = use_cuda, gamma = gamma_sdtw, normalize = use_divergence, bandwidth = bandwidth, dist_func = dist_func)

        self.block_size = block_size

    def forward(self, x : torch.tensor, x_r : torch.tensor) -> torch.tensor :
        # Compute new batch size
        virtual_batch_size = int(x.shape[0] * (x.shape[1] / self.block_size))
        
        # Reshape input tensors
        x_reshaped   = x.view(virtual_batch_size, self.block_size, -1)
        x_r_reshaped = x_r.view(virtual_batch_size, self.block_size, -1)

        block_loss = super().forward(x_reshaped, x_r_reshaped)

        # Return the loss reshaped to have the original batch size
        return block_loss.view(x.shape[0], -1).sum(dim = 1)

# def block_sdtw(x : torch.tensor, x_r : torch.tensor,
#                block_size : int, shift : int = -1, normalize_by_block_size : bool = True,
#                use_divergence : bool = False,
#                gamma_sdtw : float = 1, bandwidth : float = None, dist_func = None,
#                ) :
#     """
#     Instead of applying the dtw to the entire signal, this function applies it on block of size block_size.
#
#     @param x: (torch.tensor) First input tensor of shape B x T x 1
#     @param x_r: (torch.tensor) Second input tensor of shape B x T x 1
#     @param block_size: (int) Size of blocks into which to divide the signal.
#     @param soft_DTW_type: (int) Type of SDTW to use. 3 for standard SDTW and 4 for SDTW divergence
#
#     @return recon_error: (torch.tensor) Tensor of shape B
#     """
#
#     if shift <= 0 : shift = block_size
#
#     tmp_recon_loss = 0
#     i = 0
#     continue_cylce = True
#
#     sdtw_loss_function = SoftDTW(use_cuda = use_cuda, gamma = gamma_dtw, bandwidth = bandwidth)
#
#     while continue_cylce :
#         # Get indices for the block
#         idx_1 = int(i * block_size)
#         idx_1 = int(i * shift)
#         idx_2 = int((i + 1) * block_size) if int((i + 1) * block_size) < x.shape[1] else -1
#
#         # Get block of the signal
#         if idx_2 == -1 :
#             x_block = x[:, idx_1:, :]
#             x_r_block = x_r[:, idx_1:, :]
#         else :
#             x_block = x[:, idx_1:idx_2, :]
#             x_r_block = x_r[:, idx_1:idx_2, :]
#
#         # Compute dtw for the block
#         if soft_DTW_type == 3 : # Block SDTW
#             block_loss = dtw_loss_function(x_block, x_r_block)
#         elif soft_DTW_type == 4 : # Block SDTW divergence
#             dtw_xy_block = dtw_loss_function(x_block, x_r_block)
#             dtw_xx_block = dtw_loss_function(x_block, x_block)
#             dtw_yy_block = dtw_loss_function(x_r_block, x_r_block)
#             block_loss = dtw_xy_block - 0.5 * (dtw_xx_block + dtw_yy_block)
#
#         # (Optional) Normalize by the number of samples in the block
#         if normalize_by_block_size : block_loss = block_loss / x_block.shape[1]
#
#         # End the cylce at the last block
#         if idx_2 == -1 : continue_cylce = False
#
#         tmp_recon_loss += block_loss
#
#         # Increase index
#         i += 1
#
#     return tmp_recon_loss
#
# def block_sdtw_optimized(x : torch.tensor, x_r : torch.tensor,
#                dtw_loss_function,
#                block_size : int, soft_DTW_type : int, shift : int = -1,
#                normalize_by_block_size : bool = True):
#     """
#     Used only if length_signal % block_size == 0, i.e. the block size divide without rest the length of the signal.
#
#     @param x: (torch.tensor) First input tensor of shape B x T x 1
#     @param x_r: (torch.tensor) Second input tensor of shape B x T x 1
#     """
#     
#     if x.shape[1] % block_size != 0 :
#         raise ValueError(f"block_sdtw_optimize can be used only if length_signal % block_size == 0. Current values are length_signal = {x.shape[1]}, block_size = {block_size}. Currently length_signal % block_size = {x.shape[1] % block_size}")
#     
#     # Compute new batch size
#     virtual_batch_size = int(x.shape[0] * (x.shape[1] / block_size))
#     
#     # Reshape input tensors
#     x_reshaped   = x.view(virtual_batch_size, block_size, -1)
#     x_r_reshaped = x_r.view(virtual_batch_size, block_size, -1)
#
#     if soft_DTW_type == 3 : # Block SDTW
#         block_loss = dtw_loss_function(x_reshaped, x_r_reshaped)
#     elif soft_DTW_type == 4 : # Block SDTW divergence
#         dtw_xy_block = dtw_loss_function(x_reshaped, x_r_reshaped)
#         dtw_xx_block = dtw_loss_function(x_reshaped, x_reshaped)
#         dtw_yy_block = dtw_loss_function(x_r_reshaped, x_r_reshaped)
#         block_loss = dtw_xy_block - 0.5 * (dtw_xx_block + dtw_yy_block)
#
#     # (Optional) Normalize by the number of samples in the block
#     if normalize_by_block_size : block_loss = block_loss / block_size
#
#     # Return the loss reshaped to have the original batch size
#     return block_loss.view(x.shape[0], -1).sum(dim = 1)
#
