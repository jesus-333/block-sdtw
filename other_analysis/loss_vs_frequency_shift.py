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

from block_sdtw import reconstruction_loss, block_sdtw, block_sdtw_for_analysis
from soft_dtw_cuda import SoftDTW

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

n_samples = 300
block_size_list = [50, 100, 150]
normalize_by_n_of_block = False
add_noise = False

t_samples = np.linspace(0, 1, n_samples)

amplitude_1 = 20
amplitude_2 = 20
f_1 = 100
samples_1 =  2 * np.pi * f_1 * t_samples
sine_1 = torch.from_numpy(amplitude_1 * np.sin(samples_1))  

shift_2 = 0
f_2_list = np.arange(30) + f_1

plot_config = dict(
    figsize = (13, 10),
    fontsize = 20,
    save_fig = True,
)

if add_noise :
    noise = torch.randn(len(samples_1))
    sine_1 += noise

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Compute loss values

loss_values_per_block_size  = []
average_loss_per_block_list  = []
std_loss_per_block_list  = []
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

# Add dimensions required by loss function and loss function
sine_1 = sine_1.unsqueeze(0).unsqueeze(-1).to(device)
sdtw_loss_function = SoftDTW(use_cuda = True if device == 'cuda' else False, gamma = config_loss['gamma_dtw'])

# Variables to store the results
loss_values_per_block_size = np.zeros((len(block_size_list), len(f_2_list)))
loss_values_sdtw = []

# Compute loss for block-SDTW
config_loss['recon_loss_type'] = 4
for i in range(len(block_size_list)) :
    block_size = block_size_list[i]
    config_loss['block_size'] = block_size
    config_loss['shift'] = config_loss['block_size']
    print("---------------------------------------------------------")
    print("Compute loss for block size : {}".format(block_size))

    for j in range(len(f_2_list)) :
        f_2 = f_2_list[j]
        samples_2 =  2 * np.pi * f_2 * t_samples + shift_2
        sine_2 = torch.from_numpy(amplitude_2 * np.sin(samples_2))
        sine_2 = sine_2.unsqueeze(0).unsqueeze(-1).to(device)

        print("\tf1 = {}Hz\tf2 = {}Hz".format(f_1, f_2))

        # loss_function = reconstruction_loss(config_loss)
        tmp_loss, block_values = block_sdtw_for_analysis(sine_1, sine_2, sdtw_loss_function,config_loss['block_size'], config_loss['recon_loss_type'], config_loss['shift'], config_loss['normalize_by_block_size'])
        if normalize_by_n_of_block : tmp_loss /= len(block_values)

        loss_values_per_block_size[i, j] = tmp_loss

# Compute loss for SDTW
config_loss['recon_loss_type'] = 2
loss_function_normal_sdtw = reconstruction_loss(config_loss)
for i in range(len(f_2_list)) :
    f_2 = f_2_list[i]
    samples_2 =  2 * np.pi * f_2 * t_samples + shift_2
    sine_2 = torch.from_numpy(amplitude_2 * np.sin(samples_2))
    sine_2 = sine_2.unsqueeze(0).unsqueeze(-1).to(device)  

    normal_sdtw_value = np.ones(len(loss_values_per_block_size  )) * float(loss_function_normal_sdtw.compute_loss(sine_1, sine_2).cpu().numpy())

    loss_function = reconstruction_loss(config_loss)
    tmp_loss = float(loss_function.compute_loss(sine_2, sine_1).cpu().numpy())
    loss_values_sdtw.append(tmp_loss)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Plot config

fig, ax = plt.subplots(1, 1, figsize = plot_config['figsize'])

# Plot the loss values
for i in range(len(block_size_list)) :
    ax.plot(f_2_list, loss_values_per_block_size[i], label = "Block-SDTW - block size = {}".format(block_size_list[i]))

ax.plot(f_2_list, loss_values_sdtw, label = "SDTW", color = 'black')

ax.set_xlabel("$f_2$ [Hz]", fontsize = plot_config['fontsize'])
ax.set_ylabel("Loss value", fontsize = plot_config['fontsize'])
ax.set_xlim([f_2_list[0], f_2_list[-1]])
ax.tick_params(axis = 'both', which = 'major', labelsize = plot_config['fontsize'])
ax.legend(fontsize = plot_config['fontsize'])
ax.grid(True)
if add_noise :
    ax.set_title("$f_1$ = {}Hz, $A_1$ = {} (Noisy), $A_2$ = {}".format(f_1, amplitude_1, amplitude_2), fontsize = plot_config['fontsize'])
else :
    ax.set_title("$f_1$ = {}Hz, $A_1$ = {}, $A_2$ = {}".format(f_1, amplitude_1, amplitude_2), fontsize = plot_config['fontsize'])

fig.tight_layout()
fig.show()


if plot_config['save_fig'] :
    path_save = 'Results/loss_vs_frequency_shift/'

    os.makedirs(path_save, exist_ok = True)
    fig.savefig(path_save + 'f1_{}Hz_A1_{}_f2_{}Hz_A2_{}.png'.format(f_1, amplitude_1, f_2, amplitude_2))
