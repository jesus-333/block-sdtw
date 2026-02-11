"""
Plot the results obtained with evaluation_6.py
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

import matplotlib.pyplot as plt
import numpy as np
import os

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Settings


name_machine = "Mac_CPU"
# name_machine = "Colab"
name_machine = "IRIS"
path_files = "Results/computation_time/script_6/{}/".format(name_machine)

t_list = (np.arange(10) + 1) * 100
block_size_list = [10, 50]
batch_size = 5

plot_config = dict(
    normalization = 0,
    figsize = (15, 8),
    fontsize = 20,
    use_millisec = True,
    add_std = False,
    use_log_scale = False,
    vmin = None,
    vmax = None,
    save_fig = True
)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Load data

loss_type_list = ['SDTW', 'block_sdtw_naive', 'block_sdtw_optimized']

label_list = []
time_matrix_mean_list = []
time_matrix_std_list = []
color_list = ['green', 'red', 'blue', 'darkred', 'darkblue']

for loss_type in loss_type_list : # Iterate over loss function variation

    for i in range(len(block_size_list)) :
        block_size = block_size_list[i]
        
        # Temporary list to save the mean and std for each sequence length for the current loss function variation
        tmp_list_mean = []
        tmp_list_std = []

        # Create label 
        label = loss_type.split('_')[-1] 
        if loss_type != 'SDTW' : label += f" ({block_size})".format(block_size)
        label_list.append(label)

        for j in range(len(t_list)) : # Iterate over sequence length
            t = t_list[j]

            # Load data
            if loss_type == 'SDTW' : file_name = path_files + f"T_{t}_batch_{batch_size}_{loss_type}_time_list_loss.npy"
            else : file_name = path_files + f"T_{t}_batch_{batch_size}_{loss_type}_block_{block_size}_time_list_loss.npy"
            time_list = np.load(file_name)
            
            # Compute mean and std
            tmp_list_mean.append(np.mean(time_list))
            tmp_list_std.append(np.std(time_list))
            
        # Save mean and std
        time_matrix_mean_list.append(tmp_list_mean)
        time_matrix_std_list .append(tmp_list_std)

        if loss_type == 'SDTW' : break # SDTW does not depend on block size, so we only need to load it once

time_matrix_mean = np.asarray(time_matrix_mean_list)
time_matrix_std = np.asarray(time_matrix_std_list)

for i in range(len(label_list)) : label_list[i] = label_list[i].replace('naive', 'BlockDTW (sequential)')
for i in range(len(label_list)) : label_list[i] = label_list[i].replace('optimized', 'BlockDTW (parallel)')

if plot_config['use_millisec'] :
    time_matrix_mean *= 1000
    time_matrix_std *= 1000

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Plot

fig, ax = plt.subplots(1, 1, figsize = plot_config['figsize'])

# Plot average time
for i in range(time_matrix_mean.shape[0]) :
    if i == 0 : marker = 's'
    elif i < time_matrix_mean.shape[0] / 2 : marker = 'o'
    else : marker = 'x'
    ax.plot(t_list, time_matrix_mean[i],
            marker = marker, label = label_list[i], markersize = 12,
            color = color_list[i])

# (OPTIONAL) Add std
if plot_config['add_std'] :
    for i in range(time_matrix_mean.shape[0]) :
        ax.fill_between(t_list, time_matrix_mean[i] - time_matrix_std[i], time_matrix_mean[i] + time_matrix_std[i], 
                        alpha = 0.3, color = color_list[i])

# Add legend, labels, etc.
ax.legend(label_list, fontsize = plot_config['fontsize'])
ax.set_xlabel("N. samples", fontsize = plot_config['fontsize'])
ax.set_xlim([t_list[0], t_list[-1]])
# if not plot_config['use_log_scale'] : ax.set_ylim([0, time_matrix_mean.max() * 1.05])
ax.set_ylim([0, 270])
# ax.set_ylim([0.4, 500])
if plot_config['normalization'] == 1 or plot_config['normalization'] == 2 :
    ax.set_ylabel("Time (normalized)", fontsize = plot_config['fontsize'])
else :
    if plot_config['use_millisec'] : ax.set_ylabel("Time (ms)", fontsize = plot_config['fontsize'])
    else : ax.set_ylabel("Time (s)", fontsize = plot_config['fontsize'])
if plot_config['use_log_scale'] : ax.set_yscale('log')
ax.grid()
ax.tick_params(axis = 'both', which = 'major', labelsize = plot_config['fontsize'])

fig.tight_layout()
fig.show()

if plot_config['save_fig'] :
    path_save = "Results/computation_time/plot/"
    os.makedirs(path_save, exist_ok = True)
    path_save = f"Results/computation_time/plot/plot_6_{name_machine}"
    if plot_config['use_log_scale'] : path_save += "_log"
    if plot_config['add_std'] : path_save += "_std"
    fig.savefig(path_save + ".png", bbox_inches = 'tight')
    fig.savefig(path_save + ".pdf", bbox_inches = 'tight')


