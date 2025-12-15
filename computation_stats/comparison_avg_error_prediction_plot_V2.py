"""
Plot the results obtained with comparison_avg_error_prediction_computation_V2.py

@author: Alberto Zancanaro (Jesus)
@organization: Luxembourg Centre for Systems Biomedicine (LCSB)
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

import matplotlib.pyplot as plt
import numpy as np
import os

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Settings

plot_config = dict(
    # Info about the data used for training and prediction
    use_z_score_normalization = True,    # If True a z-score normalization will be applied signal by signal within each dataset
    portion_of_signals_for_input = 0.85, # Portion of the signals to use for training (the rest will be used for prediction)
    n_samples_to_predict = 100,            # Number of samples to predict (If negative it is ignored and the portion_of_signals_for_input is used to define the number of samples to predict. Otherwise, this parameter override portion_of_signals_for_input)
    epoch = -1,
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    clip_results = False,               # If True each row will have values 1, 2, 3 corresponding to the best, second best and third best method for each dataset
    exclude_failed_dataset = True,     # If True only the dataset where at least 1 training was successful will be plotted
    figsize = (18, 12),
    cmap = 'Greens',
    # cmap = 'seismic_r',
    aspect = 'auto',
)

loss_function_to_plot = ['MSE', 'SDTW', 'SDTW_divergence', 'pruned_SDTW', 'OTW', 'block_SDTW_10', 'block_SDTW_50']
loss_function_to_plot = ['SDTW', 'SDTW_divergence', 'pruned_SDTW', 'OTW', 'block_SDTW_10', 'block_SDTW_50']

loss_function_to_idx = dict(
    MSE = 0,
    SDTW = 1,
    SDTW_divergence = 2,
    pruned_SDTW = 3,
    OTW = 4,
    block_SDTW_10 = 5,
    block_SDTW_50 = 6
)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Load data

path_UCR_folder = "./data/UCRArchive_2018/"

# Get all the folders inside path_UCR_folder
list_all_dataset_name = [f for f in os.listdir(path_UCR_folder) if os.path.isdir(os.path.join(path_UCR_folder, f))]

if plot_config['n_samples_to_predict'] > 0 :
    folder_name = f"neurons_256_predict_samples_{plot_config['n_samples_to_predict']}"
else :
    folder_name = f"neurons_256_predict_portion_{int(plot_config['portion_of_signals_for_input'] * 100)}"

if plot_config['use_z_score_normalization'] : folder_name += '_z_score'

path_errors_matrix_folder = f"saved_model/{folder_name}/0_comparison/"

if plot_config['epoch'] == -1 : plot_config['epoch'] = 'end'

# Load matrix with the average errors for all loss functions
matrix_with_all_data_train = np.load(path_errors_matrix_folder + f"average_scores_matrix_train_{plot_config['epoch']}.npy")
matrix_with_all_data_test  = np.load(path_errors_matrix_folder + f"average_scores_matrix_test_{plot_config['epoch']}.npy")

if plot_config['exclude_failed_dataset'] :
    n_dataset_to_plot = np.sum([1 for i in range(len(list_all_dataset_name)) if matrix_with_all_data_train[i, :].sum() > 0])
else :
    n_dataset_to_plot = len(list_all_dataset_name)

matrix_to_plot_train = np.zeros((n_dataset_to_plot, len(loss_function_to_plot)))
matrix_to_plot_test = np.zeros((n_dataset_to_plot, len(loss_function_to_plot)))

# Fill the matrix to plot
idx_row_to_plot = 0
for i in range(len(list_all_dataset_name)) :
    if plot_config['exclude_failed_dataset'] :
        # Skip the iteration for datasets where all trainings failed
        if matrix_with_all_data_train[i, :].sum() == 0 :
            continue
    idx_row_original_data = i
    for j in range(len(loss_function_to_plot)) :
        idx_column_original_data = loss_function_to_idx[loss_function_to_plot[j]]
        matrix_to_plot_train[idx_row_to_plot, j] = matrix_with_all_data_train[idx_row_original_data, idx_column_original_data]
        matrix_to_plot_test[idx_row_to_plot, j]  = matrix_with_all_data_test[idx_row_original_data, idx_column_original_data]
    
    # Increment the row index for the matrix to plot
    # Note that this line is only reached if the dataset was not skipped
    # So it is reach every iteration if exclude_failed_dataset is False
    # Otherwise it is only reached for successful datasets
    idx_row_to_plot += 1


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# (OPTIONAL) Clip results

if plot_config['clip_results'] :
    # For each row, set the minimum value to 0, the middle value to 1 and the maximum value to 2
    for i in range(matrix_to_plot_train.shape[0]) :
        row_train = matrix_to_plot_train[i, :]
        row_test  = matrix_to_plot_test[i, :]

        if np.sum(row_train != 0) > 0 :
            # Note that if the sum of the row is 0 it means that the dataset was skipped during training/testing
            sorted_indices_train = np.argsort(row_train)
            for j in range(len(sorted_indices_train)) : matrix_to_plot_train[i, sorted_indices_train[j]] = j + 1

        if np.sum(row_test != 0) > 0 :
            # Note that if the sum of the row is 0 it means that the dataset was skipped during training/testing
            sorted_indices_test = np.argsort(row_test)
            for j in range(len(sorted_indices_test)) : matrix_to_plot_test[i, sorted_indices_test[j]] = j + 1

    vmin_train, vmin_test = 1, 1
    vmax_train, vmax_test = len(loss_function_to_plot), len(loss_function_to_plot)
else :
    vmin_train, vmin_test = 0, 0
    vmin_train, vmin_test = 0.5, 0.5
    vmax_train, vmax_test = 1, 1

    # vmin_train, vmin_test = 0, 0
    # vmax_train, vmax_test = 0.5, 0.5

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Plotting

fig, axs = plt.subplots(2, 1, figsize = plot_config['figsize'])

# Plot matrices
axs[0].imshow(matrix_to_plot_train.T, cmap = plot_config['cmap'], aspect = plot_config['aspect'],
              vmax = vmax_train, vmin = vmin_train
              )
axs[1].imshow(matrix_to_plot_test.T, cmap = plot_config['cmap'], aspect = plot_config['aspect'],
              vmax = vmax_test, vmin = vmin_test
              )

for i in range(2) :
    axs[i].set_xticks(np.arange(n_dataset_to_plot) + 0.5, minor = True,)
    axs[i].set_yticks(np.arange(len(loss_function_to_idx) - 1) + 0.5, minor = True)
    axs[i].set_xticklabels([], minor = True)
    axs[i].grid(which = 'minor', color = 'k', linestyle = '-', snap = False)
    # axs[i].grid(which = 'major',color = 'k', linestyle = '-', snap = False)
    # axs[i].set_xticklabels(list_all_dataset_name, rotation = 90)

    axs[i].set_yticks(np.arange(len(loss_function_to_plot)), minor = False)
    axs[i].set_yticklabels(loss_function_to_plot)
    axs[i].set_title('Train set' if i == 0 else 'Test set')

    axs[i].set_xlabel('Datasets')

    # Add colorbar
    cbar = fig.colorbar(axs[i].images[0], ax = axs[i], fraction = 0.046, pad = 0.04)
    cbar.vmin = 0.5

    

# Set x-ticks only for datasets that were used (i.e., rows that have at least one non-zero value)
xticks_train = [i for i in range(n_dataset_to_plot) if matrix_to_plot_train[i, :].sum() > 0]
xticks_test  = [i for i in range(n_dataset_to_plot) if matrix_to_plot_test[i, :].sum() > 0]
xticks_labels_train = [f"{list_all_dataset_name[i][0:3]} ({i})" for i in xticks_train]
xticks_labels_test  = [f"{list_all_dataset_name[i][0:3]} ({i})" for i in xticks_test]

axs[0].set_xticks(xticks_train)
axs[0].set_xticklabels(xticks_labels_train, rotation = 90)
axs[1].set_xticks(xticks_test)
axs[1].set_xticklabels(xticks_labels_test, rotation = 90)

# Add overall title
if plot_config['n_samples_to_predict'] > 0 : 
    overall_title = f"Average score (predicting {plot_config['n_samples_to_predict']} samples)\n"
else :
    overall_title = f"Average score (predicting portion {round((1 - plot_config['portion_of_signals_for_input']) * 100)}% of each signal)\n"
if plot_config['clip_results'] : overall_title += " (clipped)"
fig.suptitle(overall_title, fontsize = 16)

# Adjust layout and show plot
fig.tight_layout()
plt.show()

print("N. of valid dataset (TRAIN) :", len(xticks_train))
print("N. of valid dataset (TEST)  :", len(xticks_test))

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Save plot

if plot_config['n_samples_to_predict'] > 0 :
    folder_name = f"neurons_256_predict_samples_{plot_config['n_samples_to_predict']}"
else :
    folder_name = f"neurons_256_predict_portion_{int(plot_config['portion_of_signals_for_input'] * 100)}"

if plot_config['use_z_score_normalization'] : folder_name += '_z_score'

path_save_plot = f"saved_model/{folder_name}/0_comparison/"

if plot_config['clip_results'] :
    path_save_plot += "clipped_results_"

fig.savefig(path_save_plot + f"comparison_average_score_prediction_{plot_config['epoch']}.png")






