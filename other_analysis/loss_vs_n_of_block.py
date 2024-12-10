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
block_size_list = np.arange(1, 31) * 5

t_samples = np.linspace(0, 1, n_samples)

amplitude_1 = 15
f_1 = 17
samples_1 =  2 * np.pi * f_1 * t_samples
sine_1 = torch.from_numpy(amplitude_1 * np.sin(samples_1)) 

amplitude_2 = 15
f_2 = 17
shift_2 = 0.1
samples_2 =  2 * np.pi * f_2 * t_samples + shift_2
sine_2 = torch.from_numpy(amplitude_2 * np.sin(samples_2))

plot_config = dict(
    figsize = (13, 10),
    fontsize = 20,
    save_fig = True,
)

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

# Add dimensions required by loss function
sine_1 = sine_1.unsqueeze(0).unsqueeze(-1).to(device)
sine_2 = sine_2.unsqueeze(0).unsqueeze(-1).to(device)

sdtw_loss_function = SoftDTW(use_cuda = True if device == 'cuda' else False, gamma = config_loss['gamma_dtw'])

# Compute loss for block-SDTW
for i in range(len(block_size_list)) :
    block_size = block_size_list[i]
    config_loss['block_size'] = block_size
    config_loss['shift'] = config_loss['block_size']
    print("---------------------------------------------------------")
    print("Compute loss for block size : {}".format(block_size))

    # loss_function = reconstruction_loss(config_loss)
    # tmp_loss = float(loss_function.compute_loss(sine_2, sine_1).cpu().numpy())
    tmp_loss, block_values = block_sdtw_for_analysis(sine_1, sine_2, sdtw_loss_function,config_loss['block_size'], config_loss['recon_loss_type'], config_loss['shift'], config_loss['normalize_by_block_size'])

    tmp_loss /= len(block_values)
    print(np.round(block_values, 2))

    loss_values_per_block_size.append(tmp_loss.cpu().numpy())
    average_loss_per_block_list.append(np.mean(block_values))
    std_loss_per_block_list.append(np.std(block_values))
    
# Compute standard SDTW loss
config_loss['recon_loss_type'] = 1
loss_function_normal_sdtw = reconstruction_loss(config_loss)
normal_sdtw_value = np.ones(len(loss_values_per_block_size  )) * float(loss_function_normal_sdtw.compute_loss(sine_1, sine_2).cpu().numpy())


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Plot stuff

fig_loss, ax = plt.subplots(figsize=plot_config['figsize'])

ax.plot(block_size_list, loss_values_per_block_size  , label = 'Block-SDTW', color = 'black')
ax.plot(block_size_list, normal_sdtw_value, label = 'SDTW', color = 'red')

ax.set_xlim([block_size_list[0], block_size_list[-1]])
ax.set_xlabel("Block size", fontsize = plot_config['fontsize'])
ax.set_ylabel("Loss value", fontsize = plot_config['fontsize'])
ax.tick_params(axis = 'both', which = 'major', labelsize = plot_config['fontsize'])
ax.legend(fontsize = plot_config['fontsize'])
ax.grid(True)

# ax_2 = ax.twinx()
# ax_2.plot(block_size_list, average_loss_per_block_list, label = 'Average error per block', color = 'green')
# ax_2.tick_params(axis='y', labelcolor = 'green')

fig_loss.tight_layout()
fig_loss.show()

fig_signal, ax = plt.subplots(figsize=plot_config['figsize'])
ax.plot(t_samples, sine_1.squeeze().cpu().numpy(), label = 'Sine 1 (A = {} f = {}Hz)'.format(amplitude_1, f_1))
ax.plot(t_samples, sine_2.squeeze().cpu().numpy(), label = 'Sine 2 (A = {},f = {}Hz, shift = {})'.format(amplitude_1, f_2, shift_2))
ax.legend(fontsize = plot_config['fontsize'])
ax.grid(True)
ax.set_xlim([t_samples[0], t_samples[-1]])
ax.set_xlabel("Time [s]", fontsize = plot_config['fontsize'])
ax.set_ylabel("Amplitude", fontsize = plot_config['fontsize'])
ax.tick_params(axis = 'both', which = 'major', labelsize = plot_config['fontsize'])
fig_signal.tight_layout()
fig_signal.show()

if plot_config['save_fig'] :
    path_save = 'Results/loss_vs_n_of_block/sine/f1_{}Hz_A1_{}_f2_{}Hz_A2_{}_shift2_{}/'.format(f_1, amplitude_1, f_2, amplitude_2, shift_2)
    os.makedirs(path_save, exist_ok = True)

    fig_loss.savefig(path_save + 'loss_comparison.png')
    fig_signal.savefig(path_save + 'signal_comparison.png')
