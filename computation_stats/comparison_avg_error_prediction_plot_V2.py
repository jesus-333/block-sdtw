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
    sort_in_alphabetical_order = True, # If True the datasets will be sorted in alphabetical order. Note that the original order depends on the reading order from the functions in os library
    add_cross_failed_dataset = True,  # If True add a cross to the cells corresponding to failed dataset
    split_train_test_figure = True,              # If True create two different figures for train and test. Otherwise create a single figure with two subplots
    figsize = (20, 8),
    # cmap = 'Greens',
    cmap = 'Spectral',
    fontsize = 16,
    aspect = 'auto',
    save_plot = True,
    format_to_save_plot = ['png', 'pdf']
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

loss_function_to_label = dict(
    MSE = 'MSE',
    SDTW = 'SDTW',
    SDTW_divergence = 'SDTW\nDivergence',
    pruned_SDTW = 'Pruned\nSDTW',
    OTW = 'OTW',
    block_SDTW_10 = 'Block DTW\n(10)',
    block_SDTW_50 = 'Block DTW\n(50)'
)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Load data

path_UCR_folder = "./data/UCRArchive_2018/"

# Get all the folders inside path_UCR_folder
list_all_dataset_name = [f for f in os.listdir(path_UCR_folder) if os.path.isdir(os.path.join(path_UCR_folder, f))]
list_filtered_dataset = [] # List with all dataset plotted. It is the equals to list_all_dataset_name if exclude_failed_dataset is False.
list_idx_filtered_dataset = [] # List with the indices of the dataset. Each element correspond to the index in list_filtered_dataset

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

# Sort in alphabetical order
if plot_config['sort_in_alphabetical_order'] :
    idx_sort_alphabetical = np.argsort(list_all_dataset_name)
    list_all_dataset_name = np.asarray(list_all_dataset_name)[idx_sort_alphabetical]
    matrix_with_all_data_train = matrix_with_all_data_train[idx_sort_alphabetical]
    matrix_with_all_data_test = matrix_with_all_data_test[idx_sort_alphabetical]

# Count the number of dataset to plot
if plot_config['exclude_failed_dataset'] :
    n_dataset_to_plot = np.sum([1 for i in range(len(list_all_dataset_name)) if matrix_with_all_data_train[i, :].sum() > 0])
else :
    n_dataset_to_plot = len(list_all_dataset_name)

# Matrices to store the data of the dataset to plot
matrix_to_plot_train = np.zeros((n_dataset_to_plot, len(loss_function_to_plot)))
matrix_to_plot_test = np.zeros((n_dataset_to_plot, len(loss_function_to_plot)))

# Fill the matrix to plot
idx_row_to_plot = 0
for i in range(len(list_all_dataset_name)) :
    # (OPTIONAL) Skip the dataset with failed training runs (or with training runs impossible to do due to signal to short or similar)
    if plot_config['exclude_failed_dataset'] :
        # Skip the iteration for datasets where all training runs failed
        if matrix_with_all_data_train[i, :].sum() == 0 :
            continue
    
    # Add data to the matrix to plot
    # Note that this line is only reached if the dataset was not skipped
    idx_row_original_data = i
    list_filtered_dataset.append(list_all_dataset_name[i])
    list_idx_filtered_dataset.append(idx_row_original_data + 1) # +1 to have indices starting from 1 instead of 0 (more user friendly for paper readers)
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
# Plot function

def create_plot(fig, ax, matrix_to_plot, n_dataset_to_plot, vmax, vmin, title = None) :
    """
    Create a plot with the given matrix and configuration.
    Note that the function will not work if you copy and paste it in another script. It used some variable defined above
    """

    # Add crosses for failed dataset
    if plot_config['add_cross_failed_dataset'] :
        for i in range(matrix_to_plot.shape[0]) :
            for j in range(matrix_to_plot.shape[1]) :
                if matrix_to_plot[i, j] == 0 :
                    marker_size = 10 if plot_config['exclude_failed_dataset'] else 5
                    ax.plot(i, j, marker = 'x', color = 'red', markersize = marker_size, markeredgewidth = 2)

    # Set x-ticks only for datasets that were used (i.e., rows that have at least one non-zero value)
    xticks = [i for i in range(n_dataset_to_plot) if matrix_to_plot[i, :].sum() > 0]
    xticks_labels = [f"{list_filtered_dataset[i][0:3]} ({list_idx_filtered_dataset[i]})" for i in xticks]

    matrix_to_plot[matrix_to_plot == 0] = np.nan # Set the values corresponding to failed dataset to NaN to have them in white color in the plot

    # Plot matrices
    ax.imshow(matrix_to_plot.T, cmap = plot_config['cmap'], aspect = plot_config['aspect'],
              vmax = vmax, vmin = vmin
              )

    ax.set_xticks(np.arange(n_dataset_to_plot) + 0.5, minor = True,)
    ax.set_yticks(np.arange(len(loss_function_to_idx) - 1) + 0.5, minor = True)
    ax.set_xticklabels([], minor = True, fontsize = plot_config['fontsize'])
    ax.grid(which = 'minor', color = 'k', linestyle = '-', snap = False)
    # ax.grid(which = 'major',color = 'k', linestyle = '-', snap = False)
    # ax.set_xticklabels(list_all_dataset_name, rotation = 90)

    ax.set_yticks(np.arange(len(loss_function_to_plot)), minor = False)
    ax.set_yticklabels([loss_function_to_label[loss_function] for loss_function in loss_function_to_plot], minor = False, fontsize = plot_config['fontsize'])
    if title is not None : ax.set_title(title, fontsize = plot_config['fontsize'] + 2)

    ax.set_xlabel('Datasets', fontsize = plot_config['fontsize'])

    # Add colorbar
    cbar = fig.colorbar(ax.images[0], ax = ax, fraction = 0.046, pad = 0.04)
    cbar.vmin = 0.5
    cbar.vmax = 1
    cbar.ax.tick_params(labelsize = plot_config['fontsize'])

    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks_labels, rotation = 90, fontsize = plot_config['fontsize'])

    # Add overall title
    if plot_config['n_samples_to_predict'] > 0 :
        overall_title = f"Average score (predicting {plot_config['n_samples_to_predict']} samples)\n"
    else :
        overall_title = f"Average score (predicting portion {round((1 - plot_config['portion_of_signals_for_input']) * 100)}% of each signal)\n"
    if plot_config['clip_results'] : overall_title += " (clipped)"
    # fig.suptitle(overall_title, fontsize = 16, fontsize = plot_config['fontsize'] + 3)

    # Adjust layout and show plot
    fig.tight_layout()

    # print("N. of valid dataset (TRAIN) :", len(xticks_train))
    # print("N. of valid dataset (TEST)  :", len(xticks_test))

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Plot train and test results

if plot_config['split_train_test_figure'] :
    fig_train, ax_train = plt.subplots(figsize = plot_config['figsize'])
    create_plot(fig_train, ax_train, matrix_to_plot_train, n_dataset_to_plot, vmax_train, vmin_train, title = None)

    fig_test, ax_test = plt.subplots(figsize = plot_config['figsize'])
    create_plot(fig_test, ax_test, matrix_to_plot_test, n_dataset_to_plot, vmax_test, vmin_test, title = None)
else :
    fig, axs = plt.subplots(2, 1, figsize = plot_config['figsize'])
    create_plot(fig, axs[0], matrix_to_plot_train, n_dataset_to_plot, vmax_train, vmin_train, title = "Train")
    create_plot(fig, axs[1], matrix_to_plot_test, n_dataset_to_plot, vmax_test, vmin_test, title = "Test")

plt.show()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Save plot

if plot_config['save_plot'] :

    if plot_config['n_samples_to_predict'] > 0 :
        folder_name = f"neurons_256_predict_samples_{plot_config['n_samples_to_predict']}"
    else :
        folder_name = f"neurons_256_predict_portion_{int(plot_config['portion_of_signals_for_input'] * 100)}"

    if plot_config['use_z_score_normalization'] : folder_name += '_z_score'

    path_save_plot = f"saved_model/{folder_name}/0_comparison/comparison_average_score_prediction/"

    os.makedirs(path_save_plot, exist_ok = True)

    if plot_config['clip_results'] :
        path_save_plot += "clipped_results_"

    if plot_config['exclude_failed_dataset'] :
        path_save_plot += "only_successful_dataset_"

    if plot_config['split_train_test_figure'] :
        for format_to_save in plot_config['format_to_save_plot'] :
            filename_train = f"comparison_average_score_prediction_train_{plot_config['epoch']}_{plot_config['cmap']}.{format_to_save}"
            fig_train.savefig(path_save_plot + filename_train)

            filename_test = f"comparison_average_score_prediction_test_{plot_config['epoch']}_{plot_config['cmap']}.{format_to_save}"
            fig_test.savefig(path_save_plot + filename_test)
    else :
        for format_to_save in plot_config['format_to_save_plot'] :
            filename = f"comparison_average_score_prediction_train_test_{plot_config['epoch']}_{plot_config['cmap']}.{format_to_save}"
            fig.savefig(path_save_plot + filename)






