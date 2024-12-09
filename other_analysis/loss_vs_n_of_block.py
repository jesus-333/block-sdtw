"""
Compute the value of Block-SDTW for different values of block_size.
Use two sinusoid to compute the value.
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Imports

import torch
import time
import numpy as np
import os
import matplotlib.pyplot as plt

from block_sdtw import reconstruction_loss

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

n_samples = 300
block_size_list = np.arange(1, 31) * 5

amplitude_1 = 15
samples_1 = np.arange(n_samples) 
samples_1 = np.random.permutation(samples_1)
sine_1 = torch.from_numpy(amplitude_1 * np.sin(samples_1))

amplitude_2 = 15
samples_2 = np.arange(n_samples)  
samples_2 = np.random.permutation(samples_2)
sine_2 = torch.from_numpy(amplitude_2 * np.sin(samples_2))

plot_config = dict(
    figsize = (13, 10),
    fontsize = 20,
    save_fig = True,
)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Compute loss values

block_loss_values  = []
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Loss general config
config_loss = dict(
    # Training parameters
    alpha = 1,                          # Multiplier of the reconstruction error
    recon_loss_type = 3,                # Loss function for the reconstruction (0 = L2, 1 = SDTW, 2 = SDTW-Divergence, 3 = Block-SDTW, 4 = Block-SDTW-Divergence)
    block_size = 1,
    normalize_by_block_size = False,
    edge_samples_ignored = 0,           # Ignore this number of samples during the computation of the reconstructation loss
    gamma_dtw = 1,                      # Hyperparameter of the SDTW. Control the steepness of the soft-min inside the SDTW. The closer to 0 the closer the soft-min approximate the real min
    device = device,
    # device = "cpu",
)

# Add dimensions required by loss function
sine_1 = sine_1.unsqueeze(0).unsqueeze(-1).to(device)
sine_2 = sine_2.unsqueeze(0).unsqueeze(-1).to(device)

# Compute loss for block-SDTW
for i in range(len(block_size_list)) :
    block_size = block_size_list[i]
    config_loss['block_size'] = block_size
    config_loss['shift'] = config_loss['block_size']
    print("---------------------------------------------------------")
    print("Compute loss for block size : {}".format(block_size))

    loss_function = reconstruction_loss(config_loss)
    tmp_loss = float(loss_function.compute_loss(sine_2, sine_1).cpu().numpy())
    block_loss_values.append(tmp_loss)
    
# Compute standard SDTW loss
config_loss['recon_loss_type'] = 1
loss_function_normal_sdtw = reconstruction_loss(config_loss)
normal_sdtw_value = np.ones(len(block_loss_values)) * float(loss_function_normal_sdtw .compute_loss(sine_1, sine_2).cpu().numpy())

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Plot stuff

fig, ax = plt.subplots(figsize=plot_config['figsize'])

ax.plot(block_size_list, block_loss_values, label = 'Block-SDTW')
ax.plot(block_size_list, normal_sdtw_value, label = 'SDTW')

ax.set_xlim([block_size_list[0], block_size_list[-1]])
ax.set_xlabel("Block size", fontsize = plot_config['fontsize'])
ax.set_ylabel("Loss value", fontsize = plot_config['fontsize'])
ax.tick_params(axis = 'both', which = 'major', labelsize = plot_config['fontsize'])
ax.legend(fontsize = plot_config['fontsize'])
ax.grid(True)

fig.tight_layout()
fig.show()


fig, ax = plt.subplots(figsize=plot_config['figsize'])
ax.plot(sine_1.squeeze().cpu().numpy(), label = 'Sine 1')
ax.plot(sine_2.squeeze().cpu().numpy(), label = 'Sine 2')
fig.tight_layout()
fig.show()

