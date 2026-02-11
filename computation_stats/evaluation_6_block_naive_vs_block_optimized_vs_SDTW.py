"""
Similar to evaluation 6 but it will also add the computation of SDTW
"""
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

import torch
import time
import numpy as np
import os

import block_sdtw
import soft_dtw_cuda

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

T_list = (np.arange(10) + 1) * 100
block_size_list = [10, 50]
batch_size = 5

# Other parameters
use_cuda = False
n_repetitions = 200
pc_name = "Mac_CPU"
save_results = True

config_loss = dict(
    # Training parameters
    alpha = 1,                          # Multiplier of the reconstruction error
    recon_loss_type = 3,
    block_size = -1,
    normalize_by_block_size = False,
    edge_samples_ignored = 0,           # Ignore this number of samples during the computation of the reconstructation loss
    gamma_dtw = 1,                      # Hyperparameter of the SDTW. Control the steepness of the soft-min inside the SDTW. The closer to 0 the closer the soft-min approximate the real min
)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

loss_type_list = ['SDTW', 'block_sdtw_naive', 'block_sdtw_optimized']

device = 'cuda' if torch.cuda.is_available() and use_cuda else 'cpu'

config_loss['shift'] = config_loss['block_size']
config_loss['device'] = device


def repeat_inference(x, x_r, n_repetitions : int, recon_loss_type : str, config : dict) :
    time_list_loss = []

    sdtw_loss_function = soft_dtw_cuda.SoftDTW(use_cuda = True if config_loss['device'] == 'cuda' else False)

    with torch.no_grad() :
        for i in range(n_repetitions) :

            # Compute loss
            if loss_type_to_use == 'SDTW' :
                time_start = time.time()
                _ = sdtw_loss_function(x, x_r)
                time_end = time.time()
            elif loss_type_to_use == 'block_sdtw_naive' :
                time_start = time.time()
                _ = block_sdtw.block_sdtw(x, x_r, sdtw_loss_function, config_loss['block_size'], soft_DTW_type = 3)
                time_end = time.time()
            elif loss_type_to_use == 'block_sdtw_optimized' :
                time_start = time.time()
                _ = block_sdtw.block_sdtw_optimized(x, x_r, sdtw_loss_function, config_loss['block_size'], soft_DTW_type = 3)
                time_end = time.time()
            else :
                raise ValueError(f"loss type to use must be 'block_sdtw_naive' or 'block_sdtw_optimized'. Current value is {loss_type_to_use}")

            # Save computations times
            time_list_loss.append(time_end - time_start)

    return time_list_loss

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Compute the average inference time

for loss_type_to_use in loss_type_list :
    config_loss['recon_loss_type'] = loss_type_to_use
    print("Loss : {}".format(loss_type_to_use))
    print("Batch size = {}".format(batch_size))

    for i in range(len(T_list)) : # Loop over the number of time samples
        T  = T_list[i]
        print("\ti = {} (T = {})".format(i, T))
        
        # Create synthetic data
        x   = torch.rand(batch_size, T, 1).to(device)
        x_r = torch.rand(batch_size, T, 1).to(device)

        for j in range(len(block_size_list)) :
            block_size = block_size_list[j]
            config_loss['block_size'] = block_size
            if loss_type_to_use != 'SDTW' : print("\t\tblock_size = {}".format(block_size))

            # Compute inference time
            time_list_loss  = repeat_inference(x, x_r, n_repetitions, loss_type_to_use, config_loss)

            if save_results :
                # Create the path if it does not exist
                path_save = f'Results/computation_time/script_6/{pc_name}/'
                os.makedirs(path_save, exist_ok = True)
                path_save = f'Results/computation_time/script_6/{pc_name}/T_{T}_batch_{batch_size}_{loss_type_to_use}'
                if loss_type_to_use != 'SDTW' : path_save += f'_block_{block_size}'

                # Save list in npy format
                np.save(path_save + '_time_list_loss.npy', time_list_loss)

            if loss_type_to_use == 'SDTW' : break # No need to loop over the block size for the SDTW since it does not use blocks


