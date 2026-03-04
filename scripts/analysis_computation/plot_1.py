"""
Plot the results obtained with evaluation_1.py
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Imports

import matplotlib.pyplot as plt
import numpy as np

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Settings

path_files = "Results/computation_time/raspberry/"
path_files = "Results/computation_time/script_1/PC_Lab/"

t_list = (np.arange(19) + 2) * 50
t_list = (np.arange(10) + 1) * 100

loss_type_list = [1, 3]

plot_config = dict(
    normalization = 1,
    figsize = (15, 8),
    fontsize = 20,
    add_std = True,
    use_log_scale = False,
    save_fig = False
)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Load data

label_list = []
time_matrix_mean = np.zeros((len(loss_type_list), len(t_list)))
time_matrix_std = np.zeros((len(loss_type_list), len(t_list)))

for i in range(len(loss_type_list)) : # Iterate over loss function variation
    loss_type = loss_type_list[i]
    loss_type_str = 'SDTW_rust' if loss_type == 0 else 'SDTW_standard' if loss_type == 1 else 'SDTW_divergence' if loss_type == 2 else 'SDTW_block' if loss_type == 3 else 'SDTW_block_divergence'

    label_list.append(loss_type_str)

    for j in range(len(t_list)) : # Iterate over sequence length
        t = t_list[j]

        # Load data
        file_name = path_files + "T_{}_{}_time_list_loss.npy".format(t, loss_type_str)
        time_list = np.load(file_name)
        
        # Compute mean and std
        time_matrix_mean[i, j] = np.mean(time_list)
        time_matrix_std[i, j] = np.std(time_list)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# (Optional) Normalization

if plot_config['normalization'] == 1 :
    # Normalize by the first value of each row
    time_matrix_mean /= time_matrix_mean[:, 0][:, None]

elif plot_config['normalization'] == 2 :
    # Normalize by the first value of SDTW_standard
    if 'SDTW_standard' not in label_list : print("To use normalization 2 you must have load the SDTW_standard data")
    idx_row_SDTW_standard = np.array(label_list) == "SDTW_standard"
    normalization_value = time_matrix_mean[idx_row_SDTW_standard, 0]
    time_matrix_mean /= normalization_value
else :
    print("No normalization selected")

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Plot

fig, ax = plt.subplots(1, 1, figsize = plot_config['figsize'])

# Plot average time
ax.plot(t_list, time_matrix_mean.T, marker = 'o')

# (OPTIONAL) Add std
if plot_config['add_std'] :
    for i in range(len(loss_type_list)) :
        ax.fill_between(t_list, time_matrix_mean[i] - time_matrix_std[i], time_matrix_mean[i] + time_matrix_std[i], alpha = 0.3)

# Add legend, labels, etc.
ax.legend(label_list, fontsize = plot_config['fontsize'])
ax.set_xlabel("Sequence length", fontsize = plot_config['fontsize'])
if plot_config['normalization'] == 1 or plot_config['normalization'] == 2 : ax.set_ylabel("Time (normalized)", fontsize = plot_config['fontsize'])
else : ax.set_ylabel("Time (s)", fontsize = plot_config['fontsize'])
ax.set_xlim([t_list[0], t_list[-1]])
if plot_config['use_log_scale'] : ax.set_yscale('log')
ax.grid()

fig.tight_layout()
fig.show()
