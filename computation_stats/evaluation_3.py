"""
Compute the block-DTW for various block size, given a fixed length T
"""
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Imports

import torch
import time
import numpy as np
import os

from block_sdtw import reconstruction_loss

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
block_size_list = (np.arange(20) + 1) * 5
T_list = [100, 200, 500]

# Specify the loss type to use
# 3 : use the Block version of the Soft-DTW 
# 4 : use the Block version of the Soft-DTW divergence 
loss_type_list = [3]

# Other parameters
use_cuda = False
n_average = 40
pc_name = "PC_Lab"
save_results = True

config_loss = dict(
    # Training parameters
    alpha = 1,                          # Multiplier of the reconstruction error
    recon_loss_type = 1,                # Loss function for the reconstruction (0 = L2, 1 = SDTW, 2 = SDTW-Divergence)
    block_size = 50,
    normalize_by_block_size = False,
    edge_samples_ignored = 0,           # Ignore this number of samples during the computation of the reconstructation loss
    gamma_dtw = 1,                      # Hyperparameter of the SDTW. Control the steepness of the soft-min inside the SDTW. The closer to 0 the closer the soft-min approximate the real min
    device = "cuda" if torch.cuda.is_available() else "cpu",
    # device = "cpu",
)
config_loss['shift'] = config_loss['block_size']

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

if torch.cuda.is_available() and use_cuda :
    device = 'cuda' 
else :
    device = 'cpu'

def repeat_inference(x, n_average : int, config : dict) :
    time_list_loss = []

    loss_function = reconstruction_loss(config)

    with torch.no_grad() :
        for i in range(n_average) :
            # Start time
            time_start = time.time()

            # Compute loss
            loss_value = loss_function.compute_loss(x, x)

            # Save computations times
            time_list_loss.append(time.time() - time_start)

    return time_list_loss 

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Compute the average inference time

for loss_type_to_use in loss_type_list :
    loss_type_str = 'SDTW_rust' if loss_type_to_use == 0 else 'SDTW_standard' if loss_type_to_use == 1 else 'SDTW_divergence' if loss_type_to_use == 2 else 'SDTW_block' if loss_type_to_use == 3 else 'SDTW_block_divergence'
    config_loss['recon_loss_type'] = loss_type_to_use
    print("Loss : {}".format(loss_type_str))

    for j in range(len(T_list)) : # Loop over the number of time samples
        T  = T_list[j]
        print("\tj = {} (T = {})".format(j, T))
        
        # Create synthetic data
        x = torch.rand(1, T, 1).to(device)

        for k in range(len(block_size_list)) :
            block_size = block_size_list[k]
            config_loss['block_size'] = block_size
            config_loss['shift'] = block_size
            print("\t\tblock_size = {}".format(block_size))

            # Compute inference time
            time_list_loss  = repeat_inference(x, n_average, config_loss)

            if save_results :
                # Create the path if it does not exist
                path_save = 'Results/computation_time/script_3/{}/'.format(pc_name)
                os.makedirs(path_save, exist_ok = True)
                path_save = 'Results/computation_time/script_3/{}/T_{}_{}_block_{}'.format(pc_name, T, loss_type_str, block_size)

                # Save list in npy format
                np.save(path_save + '_time_list_loss.npy', time_list_loss)
