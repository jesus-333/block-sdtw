"""
Compare the SDTW (implemented in soft_dtw_cuda.py) vs the block-SDTW, implemented inside block_sdtw.py
The block-SDTW is computed for different block size

Please note that the block-SDTW still use the SDTW implementation of soft_dtw_cuda to compute the SDTW inside the various block
"""
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

import torch
import time
import numpy as np
import os
try :
    import soft_dtw_rust
    import_soft_dtw_rust = True
except ImportError:
    import_soft_dtw_rust = False
    print("Rust implementation of the soft-DTW is not available")
    print("Please install the package from https://pypi.org/project/soft-dtw-rust/")

from block_sdtw import reconstruction_loss

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
T_list = (np.arange(40) + 1) * 50
T_list = (np.arange(20) + 1) * 50
block_size_list = [5, 10, 25, 50, 100]

# Specify the loss type to use
# 0 : use the Rust implementation of the soft-DTW
# 1 : use the CUDA implementation of the soft-DTW
# 2 : use the Soft-DTW divergence
# 3 : use the Block version of the Soft-DTW
# 4 : use the Block version of the Soft-DTW divergence
loss_type_list = [1, 3]

# Other parameters
use_cuda = True
n_repetitions = 400
pc_name = "Mac_CPU"
save_results = True

config_loss = dict(
    # Training parameters
    alpha = 1,                          # Multiplier of the reconstruction error
    recon_loss_type = 1,                # Loss function for the reconstruction (0 = L2, 1 = SDTW, 2 = SDTW-Divergence)
    block_size = 50,
    normalize_by_block_size = False,
    edge_samples_ignored = 0,           # Ignore this number of samples during the computation of the reconstructation loss
    gamma_dtw = 1,                      # Hyperparameter of the SDTW. Control the steepness of the soft-min inside the SDTW. The closer to 0 the closer the soft-min approximate the real min
    # device = "cuda",
    device = "cpu",
)
config_loss['shift'] = config_loss['block_size']

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

if 0 in loss_type_list and not import_soft_dtw_rust :
    raise ValueError("The Rust implementation of the soft-DTW is not available. Please install the package from https://pypi.org/project/soft-dtw-rust/")

device = 'cuda' if torch.cuda.is_available() and use_cuda else 'cpu'

def repeat_inference(x, n_repetitions : int, recon_loss_type : int, config : dict) :
    time_list_loss = []

    if recon_loss_type > 0 :
        loss_function = reconstruction_loss(config)

    with torch.no_grad() :
        for i in range(n_repetitions) :
            # Start time
            time_start = time.time()

            # Compute loss
            if recon_loss_type == 0 :
                loss_value = soft_dtw_rust.compute_sdtw_2d(x.squeeze().numpy().astype('float64'), x.squeeze().numpy().astype('float64'), 1)
            else :
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

        if loss_type_to_use == 3 or loss_type_to_use == 4 :
            for k in range(len(block_size_list)) :
                block_size = block_size_list[k]
                config_loss['block_size'] = block_size
                config_loss['shift'] = block_size
                print("\t\tblock_size = {}".format(block_size))

                # Compute inference time
                time_list_loss  = repeat_inference(x, n_repetitions, loss_type_to_use, config_loss)

                if save_results :
                    # Create the path if it does not exist
                    path_save = 'Results/computation_time/script_2/{}/'.format(pc_name)
                    os.makedirs(path_save, exist_ok = True)
                    path_save = 'Results/computation_time/script_2/{}/T_{}_{}_block_{}'.format(pc_name, T, loss_type_str, block_size)

                    # Save list in npy format
                    np.save(path_save + '_time_list_loss.npy', time_list_loss)
        else :
            # Compute inference time
            time_list_loss  = repeat_inference(x, n_repetitions, loss_type_to_use, config_loss)

            if save_results :

                # Create the path if it does not exist
                path_save = 'Results/computation_time/script_2/{}/'.format(pc_name)
                os.makedirs(path_save, exist_ok = True)
                path_save = 'Results/computation_time/script_2/{}/T_{}_{}'.format(pc_name, T, loss_type_str)

                # Save list in npy format
                np.save(path_save + '_time_list_loss.npy', time_list_loss)
