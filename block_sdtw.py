# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

import torch
from torch import nn
import torch.nn.functional as F

from soft_dtw_cuda import SoftDTW
from numba import jit
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

def block_sdtw(x : torch.tensor, x_r : torch.tensor, 
               dtw_loss_function, 
               block_size : int, soft_DTW_type : int, shift : int = -1,
               normalize_by_block_size : bool = True):
    """
    Instead of applying the dtw to the entire signal, this function applies it on block of size block_size.

    @param x: (torch.tensor) First input tensor of shape B x T x 1
    @param x_r: (torch.tensor) Second input tensor of shape B x T x 1
    @param dtw_loss_function: (function) The dtw implementation to use. Actually during the training I use the one provided by https://github.com/Maghoumi/pytorch-softdtw-cuda. This parameter exist to allow the use of other implementation
    @param block_size: (int) Size of blocks into which to divide the signal.
    @param soft_DTW_type: (int) Type of SDTW to use. 3 for standard SDTW and 4 for SDTW divergence

    @return recon_error: (torch.tensor) Tensor of shape B
    """

    if shift <= 0 : shift = block_size

    tmp_recon_loss = 0
    i = 0
    continue_cylce = True

    while continue_cylce :
        # Get indicies for the block
        idx_1 = int(i * block_size)
        idx_1 = int(i * shift)
        idx_2 = int((i + 1) * block_size) if int((i + 1) * block_size) < x.shape[1] else -1

        # Get block of the signal
        # Note that the order of the axis is different. Check the note in the compute_dtw_loss_along_channels function, at the beggining of the for cycle.
        x_block = x[:, idx_1:idx_2, :]
        x_r_block = x_r[:, idx_1:idx_2, :]

        # Compute dtw for the block
        if soft_DTW_type == 3 : # Standard SDTW
            block_loss = dtw_loss_function(x_block, x_r_block)
        elif soft_DTW_type == 4 : # SDTW divergence
            dtw_xy_block = dtw_loss_function(x_block, x_r_block)
            dtw_xx_block = dtw_loss_function(x_block, x_block)
            dtw_yy_block = dtw_loss_function(x_r_block, x_r_block)
            block_loss = dtw_xy_block - 0.5 * (dtw_xx_block + dtw_yy_block)

        # (Optional) Normalize by the number of samples in the block
        if normalize_by_block_size : block_loss = block_loss / (idx_2 - idx_1)

        # End the cylce at the last block
        if idx_2 == -1 : continue_cylce = False

        # Accumulate the loss for the various block
        # if continue_cylce :
        #     tmp_recon_loss += block_loss
        tmp_recon_loss += block_loss
        # print("\t", i, idx_1, idx_2, float(block_loss.mean().detach()))

        # Increase index
        i += 1

    return tmp_recon_loss

def recon_loss_mse(x, x_r):
    return F.mse_loss(x, x_r)

class reconstruction_loss():
    def __init__(self, config : dict):
        """
        Class that compute the loss function for the Variational autoencoder
        """
        # Reconstruction loss
        if config['recon_loss_type'] == 0: # L2 loss
            self.recon_loss_function = recon_loss_mse
        elif config['recon_loss_type'] == 1 or config['recon_loss_type'] == 2 or \
             config['recon_loss_type'] == 3 or config['recon_loss_type'] == 4: # SDTW/SDTW divergence/Block-SDTW/Block-SDTW-Divergence
            gamma_dtw = config['gamma_dtw'] if 'gamma_dtw' in config else 1
            use_cuda = True if config['device'] == 'cuda' else False
            config['soft_DTW_type'] = config['recon_loss_type']
            self.recon_loss_function = SoftDTW(use_cuda = use_cuda, gamma = gamma_dtw)

            # Extra parameter for the block version
            if config['recon_loss_type'] == 3 or config['recon_loss_type'] == 4 :
                self.block_size = config['block_size']
                self.shift = config['shift'] if 'config' else config['block_size']
                self.normalize_by_block_size = config['normalize_by_block_size'] if 'normalize_by_block_size' in config else False

                if self.shift > self.block_size :
                    print("Shift cannot be bigger than block size. Current values shift = {}, block_size = {}".format(self.shift, self.block_size))
                    print("Set shift = block_size")
                    self.shift = self.block_size
        else :
            raise ValueError("recon_loss_type must have an integer value between 0 and 3. Current value is {}".format(config['recon_loss_type']))
        self.recon_loss_type = config['recon_loss_type']

        self.edge_samples_ignored = config['edge_samples_ignored'] if 'edge_samples_ignored' in config else 0
        if self.edge_samples_ignored < 0: self.edge_samples_ignored = 0

        # Hyperparameter for the various part of the loss
        self.alpha = config['alpha'] if 'alpha' in config else 1 # Recon

        self.config = config

    def compute_loss(self, x, x_r):

        if self.recon_loss_type == 0 : # MSE
            tmp_recon_loss = self.recon_loss_function(x, x_r)
        elif self.recon_loss_type == 1 : # Soft-DTW
            tmp_recon_loss = self.recon_loss_function(x, x_r)
        elif self.recon_loss_type == 2 : # Soft-DTW divergence
            dtw_xy = self.recon_loss_function(x, x_r)
            dtw_xx = self.recon_loss_function(x, x)
            dtw_yy = self.recon_loss_function(x_r, x_r)
            tmp_recon_loss = dtw_xy - 0.5 * (dtw_xx + dtw_yy)
        elif self.recon_loss_type == 3 or self.recon_loss_type == 4: # Block-SDTW/Block-SDTW-Divergence
            tmp_recon_loss = block_sdtw(x, x_r, 
                                        self.recon_loss_function, 
                                        self.block_size, self.recon_loss_type, self.shift,
                                        self.normalize_by_block_size,
                                        )
        else :
            str_error = "soft_DTW_type must have one of the following values:\n"
            str_error += "\t 1 (classical SDTW)"
            str_error += "\t 1 (SDTW divergence)"
            str_error += "\t 1 (Block SDTW)"
            str_error += "\t 1 (Block-SDTW-Divergence)"
            str_error += "Current values is {}".format(self.recon_loss_type)
            raise ValueError(str_error)

        recon_loss = self.alpha * tmp_recon_loss.mean()

        return recon_loss
