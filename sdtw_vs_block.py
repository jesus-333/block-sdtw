
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

import numpy as np
import torch
import matplotlib.pyplot as plt
import os

import dataset
from block_sdtw import reconstruction_loss

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Get data

save_figure = False

# Get dataset
x_train, y_train, _, _ = dataset.get_dataset()
x_subset = dataset.get_subset(x_train, y_train, n = 60)

# Sample two random signals
idx_1 = np.random.randint(0, len(x_train))
idx_2 = np.random.randint(0, len(x_train))
# idx_1, idx_2 = 41, 60
idx_1, idx_2 = 41, 59
x_1 = torch.tensor(np.expand_dims(x_train[idx_1], 0))
x_2 = torch.tensor(np.expand_dims(x_train[idx_2], 0))

# Create noisy copy of x_1
x_1_noisy = x_1 + np.random.normal(0, 0.1, x_1.shape)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Compute losses

config = dict(
    recon_loss_type = 1,
    block_size = 10,
    shift = 5,
    edge_samples_ignored = 0,
    gamma_dtw = 1,
    device = "cpu",
)

# SDTW loss
config['recon_loss_type'] = 1
loss_function = reconstruction_loss(config)
loss_sdtw_x1_and_noisy_version = loss_function.compute_loss(x_1, x_1_noisy)
loss_sdtw_x1_and_x2 = loss_function.compute_loss(x_1, x_2)

# Block SDTW loss
config['recon_loss_type'] = 3
loss_function = reconstruction_loss(config)
loss_block_sdtw_x1_and_noisy_version = loss_function.compute_loss(x_1, x_1_noisy)
loss_block_sdtw_x1_and_x2 = loss_function.compute_loss(x_1, x_2)

# Remove extra dimensions (needed for the loss computation)
x_1 = x_1.numpy().squeeze()
x_1_noisy = x_1_noisy.numpy().squeeze()
x_2 = x_2.numpy().squeeze()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Plot results

fig, ax = plt.subplots(2, 1, figsize = (10, 10))

ax[0].plot(x_1, label = 'x_1')
ax[0].plot(x_1_noisy, label = 'x_1_noisy')

ax[1].plot(x_1, label = 'x_1')
ax[1].plot(x_2, label = 'x_2')

ax[0].set_title("SDTW loss = {:.2f} - Block loss = {:.2f}".format(loss_sdtw_x1_and_noisy_version, loss_block_sdtw_x1_and_noisy_version))
ax[1].set_title("SDTW loss = {:.2f} - Block loss = {:.2f}".format(loss_sdtw_x1_and_x2, loss_block_sdtw_x1_and_x2))

for a in ax :
    a.legend()
    a.grid()

fig.tight_layout()
fig.show()


if save_figure :
    path_save = 'Results/sdtw_vs_block/'
    os.makedirs(path_save, exist_ok = True)

    fig.savefig(path_save + 'sdtw_vs_block_' + str(idx_1) + '_' + str(idx_2) + '.png')
    plt.close()
