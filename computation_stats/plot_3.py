"""
Plot the results obtained with evaluation_3.py
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Imports

import matplotlib.pyplot as plt
import numpy as np
import os

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Settings

name_machine = "PC_Lab"
path_files = "Results/computation_time/script_3/{}/".format(name_machine)

block_size_list = (np.arange(20) + 1) * 5
t_list = [100, 200, 500]

loss_type_list = [3]

plot_config = dict(
    normalization = 0,
    figsize = (15, 8),
    fontsize = 20,
    use_millisec = True,
    add_std = False,
    use_log_scale = False,
    save_fig = True
)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Load data

label_list = []
time_matrix_mean_list = []
time_matrix_std_list = []

for i in range(len(loss_type_list)) : # Iterate over loss function variation
    loss_type = loss_type_list[i]
    loss_type_str = 'SDTW_rust' if loss_type == 0 else 'SDTW_standard' if loss_type == 1 else 'SDTW_divergence' if loss_type == 2 else 'SDTW_block' if loss_type == 3 else 'SDTW_block_divergence'

    for j in range(len(t_list)) :
        t = t_list[j]

        tmp_list_mean = []
        tmp_list_std = []
        label_list.append(loss_type_str.replace('_', ' ') + " (T = {})".format(t))
        for k in range(len(block_size_list)) : # Iterate over sequence length
            block_size = block_size_list[k]

            # Load data
            file_name = path_files + "T_{}_{}_block_{}_time_list_loss.npy".format(t, loss_type_str, block_size)
            time_list = np.load(file_name)
            
            tmp_list_mean.append(np.mean(time_list))
            tmp_list_std.append(np.std(time_list))
        
        # Compute mean and std
        time_matrix_mean_list.append(tmp_list_mean)
        time_matrix_std_list .append(tmp_list_std)

time_matrix_mean = np.asarray(time_matrix_mean_list )
time_matrix_std = np.asarray(time_matrix_std_list )

if plot_config['use_millisec'] : 
    time_matrix_mean *= 1000 
    time_matrix_std *= 1000 

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# (Optional) Normalization


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Plot

fig, ax = plt.subplots(1, 1, figsize = plot_config['figsize'])

# Plot average time
ax.plot(block_size_list, time_matrix_mean.T, marker = 'o')

# (OPTIONAL) Add std
if plot_config['add_std'] :
    for i in range(len(loss_type_list)) :
        ax.fill_between(block_size_list, time_matrix_mean[i] - time_matrix_std[i], time_matrix_mean[i] + time_matrix_std[i], alpha = 0.3)

# Add legend, labels, etc.
ax.legend(label_list, fontsize = plot_config['fontsize'])
ax.set_xlabel("Block length", fontsize = plot_config['fontsize'])
if plot_config['normalization'] == 1 or plot_config['normalization'] == 2 : ax.set_ylabel("Time (normalized)", fontsize = plot_config['fontsize'])
else : 
    if plot_config['use_millisec'] : ax.set_ylabel("Time (ms)", fontsize = plot_config['fontsize'])
    else : ax.set_ylabel("Time (s)", fontsize = plot_config['fontsize'])
ax.set_xlim([block_size_list[0], block_size_list[-1]])
if plot_config['use_log_scale'] : ax.set_yscale('log')
ax.grid()
ax.tick_params(axis = 'both', which = 'major', labelsize = plot_config['fontsize'])

fig.tight_layout()
fig.show()

if plot_config['save_fig'] :
    path_save = "Results/computation_time/plot/"
    os.makedirs(path_save, exist_ok = True)
    fig.savefig(path_save + "plot_3_{}.png".format(name_machine), bbox_inches = 'tight')
    fig.savefig(path_save + "plot_3_{}.pdf".format(name_machine), bbox_inches = 'tight')
    # fig.savefig(path_save + "plot_3_{}.eps".format(name_machine), bbox_inches = 'tight')






