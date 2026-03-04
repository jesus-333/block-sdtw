"""
Plot the results obtained with evaluation_4.py
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Imports

import matplotlib.pyplot as plt
import numpy as np
import os

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Settings


name_machine = "PC_Lab"
name_machine = "WSL_CPU"
# name_machine = "WSL_CUDA"
# name_machine = "Colab_CUDA"
path_files = "Results/computation_time/script_4/{}/".format(name_machine)

loss_type_list = [1, 3]
t_list = [500, 1000]
block_size_list = [100, 250]
batch_size_list = (np.arange(20) + 1) * 5


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

for loss_type in loss_type_list : # Iterate over loss function variation
    loss_type_str = 'SDTW_rust' if loss_type == 0 else 'SDTW_standard' if loss_type == 1 else 'SDTW_divergence' if loss_type == 2 else 'SDTW_block' if loss_type == 3 else 'SDTW_block_divergence'

    if loss_type == 3 or loss_type == 4 :
        for i in range(len(block_size_list)) :
            block_size = block_size_list[i]
            print(block_size)

            for j in range(len(t_list)) : # Iterate over sequence length
                t = t_list[j]

                label_list.append(loss_type_str.replace('_', ' ') + " (block = {}, T = {})".format(block_size, t))
                tmp_list_mean = []
                tmp_list_std = []
                for k in range(len(batch_size_list)) :
                    batch_size = batch_size_list[k]

                    # Load data
                    file_name = path_files + "T_{}_batch_{}_{}_block_{}_time_list_loss.npy".format(t, batch_size, loss_type_str, block_size)
                    time_list = np.load(file_name)
                    
                    # Compute mean and std
                    tmp_list_mean.append(np.mean(time_list))
                    tmp_list_std.append(np.std(time_list))
        
                # Save mean and std
                time_matrix_mean_list.append(tmp_list_mean)
                time_matrix_std_list .append(tmp_list_std)

    else :
        for j in range(len(t_list)) : # Iterate over sequence length
            t = t_list[j]
            label_list.append(loss_type_str.replace('_', ' ') + '(T = {})'.format(t))

            tmp_list_mean = []
            tmp_list_std = []
            for k in range(len(batch_size_list)) :
                batch_size = batch_size_list[k]

                # Load data
                file_name = path_files + "T_{}_batch_{}_{}_time_list_loss.npy".format(t, batch_size, loss_type_str)
                time_list = np.load(file_name)

                # Compute mean and std
                tmp_list_mean.append(np.mean(time_list))
                tmp_list_std.append(np.std(time_list))
            
            # Save mean and std
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
ax.plot(batch_size_list, time_matrix_mean.T, marker = 'o')

# (OPTIONAL) Add std
if plot_config['add_std'] :
    for i in range(len(loss_type_list)) :
        ax.fill_between(batch_size_list, time_matrix_mean[i] - time_matrix_std[i], time_matrix_mean[i] + time_matrix_std[i], alpha = 0.3)

# Add legend, labels, etc.
ax.legend(label_list, fontsize = plot_config['fontsize'])
ax.set_xlabel("Batch size", fontsize = plot_config['fontsize'])
ax.set_xlim([batch_size_list[0], batch_size_list[-1]])
if plot_config['normalization'] == 1 or plot_config['normalization'] == 2 : ax.set_ylabel("Time (normalized)", fontsize = plot_config['fontsize'])
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
    path_save = "Results/computation_time/plot/plot_4_{}".format(name_machine)
    if plot_config['use_log_scale'] : path_save += "_log"
    fig.savefig(path_save + ".png", bbox_inches = 'tight')
    fig.savefig(path_save + ".pdf", bbox_inches = 'tight')
    # fig.savefig(path_save + "plot_3_{}.png".format(name_machine), bbox_inches = 'tight')


