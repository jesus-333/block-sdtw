"""
Compare the prediction of model trained with SDTW, Pruned SDTW and Block SDTW.
For each dataset the average prediction error is computed.

To compute the average prediction error we follow these steps:
- Train the model with SDTW/Pruned SDTW/Block SDTW (script train_V3_UCR_timeseries.py) (The training is done with raw data)
- Load the specific dataset and load the weights of the model trained with SDTW/Pruned SDTW/Block SDTW on the dataset
- Compute the signal prediction on the test set
- Normalize the predicted signal and the ground truth signal between 0 and 1
- Compare the signal prediction with the ground truth using the standard DTW function for all 3 methods.

Notes :
- The reason to use standard DTW to compare the predictions is due to the fact that the 3 loss functions (SDTW, Pruned SDTW, Block SDTW) have different numerical ranges as output.
- The use of a 4th function (standard DTW) to compare the predictions allows to have a fair comparison between the 3 methods.
- Note that each prediction is normalized on the number of samples in the signal. This allow to have an average error per sample

"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

import dtw
import matplotlib.pyplot as plt
import numpy as np
import os
import torch

import dataset
from model import MultiLayerPerceptron

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Settings

plot_config = dict(
    # Info about the data used for training and prediction
    portion_of_signals_for_input = 0.85, # Portion of the signals to use for training (the rest will be used for prediction)
    n_samples_to_predict = 100,            # Number of samples to predict (If negative it is ignored and the portion_of_signals_for_input is used to define the number of samples to predict. Otherwise, this parameter override portion_of_signals_for_input)
    block_size = 10,
    normalize_0_1_range = True,        # Normalize each signal between 0 and 1 before DTW computation
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    clip_results = False,               # If True each row will have values 1, 2, 3 corresponding to the best, second best and third best method for each dataset
    figsize = (18, 12),
    cmap = 'Reds',
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

path_errors_matrix_folder = f"saved_model/{folder_name}/0_comparison/"

if plot_config['normalize_0_1_range'] :
    path_errors_matrix_folder += "normalized_"

# Load matrix with the average errors for all loss functions
matrix_with_data_to_plot_train = np.load(path_errors_matrix_folder + "average_errors_matrix_train.npy")
matrix_with_data_to_plot_test  = np.load(path_errors_matrix_folder + "average_errors_matrix_test.npy")

# Create matrix to store the average errors for the selected loss functions
matrix_to_plot_train = np.zeros((len(list_all_dataset_name), len(loss_function_to_plot)))
matrix_to_plot_test = np.zeros((len(list_all_dataset_name), len(loss_function_to_plot)))

# Fill the matrix to plot
for i in range(len(loss_function_to_plot)) :
    idx = loss_function_to_idx[loss_function_to_plot[i]]
    matrix_to_plot_train[:, i] = matrix_with_data_to_plot_train[:, idx]
    matrix_to_plot_test[:, i]  = matrix_with_data_to_plot_test[:, idx]

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

            # matrix_to_plot_train[i, sorted_indices_train[0]] = 1
            # matrix_to_plot_train[i, sorted_indices_train[1]] = 2
            # matrix_to_plot_train[i, sorted_indices_train[2]] = 3

        if np.sum(row_test != 0) > 0 :
            # Note that if the sum of the row is 0 it means that the dataset was skipped during training/testing
            sorted_indices_test = np.argsort(row_test)
            for j in range(len(sorted_indices_test)) : matrix_to_plot_test[i, sorted_indices_test[j]] = j + 1
            # matrix_to_plot_test[i, sorted_indices_test[0]] = 1
            # matrix_to_plot_test[i, sorted_indices_test[1]] = 2
            # matrix_to_plot_test[i, sorted_indices_test[2]] = 3

    vmax_train = len(loss_function_to_plot)
    vmax_test  = len(loss_function_to_plot)
else :
    vmax_train = np.mean(matrix_to_plot_train[matrix_to_plot_train != 0]) + np.std(matrix_to_plot_train[matrix_to_plot_train != 0])
    vmax_test  = np.mean(matrix_to_plot_test[matrix_to_plot_test != 0]) + np.std(matrix_to_plot_test[matrix_to_plot_test != 0])

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Plotting

fig, axs = plt.subplots(2, 1, figsize = plot_config['figsize'])

# Plot matrices
axs[0].imshow(matrix_to_plot_train.T, cmap = plot_config['cmap'], aspect = plot_config['aspect'],
              vmax = vmax_train, vmin = 0
              )
axs[1].imshow(matrix_to_plot_test.T, cmap = plot_config['cmap'], aspect = plot_config['aspect'],
              vmax = vmax_test, vmin = 0
              )

for i in range(2) :
    axs[i].set_xticks(np.arange(len(list_all_dataset_name)) + 0.5, minor = True,)
    # axs[i].set_yticks(np.arange(3) + 0.5, minor = True)
    axs[i].set_xticklabels([], minor = True)
    axs[i].grid(which = 'minor', color = 'w', linestyle = '-', snap = False)
    # axs[i].set_xticklabels(list_all_dataset_name, rotation = 90)

    axs[i].set_yticks(np.arange(len(loss_function_to_plot)), minor = False)
    axs[i].set_yticklabels(loss_function_to_plot)
    axs[i].set_title('Train set' if i == 0 else 'Test set')

    axs[i].set_xlabel('Datasets')

xticks_train = [i for i in range(len(list_all_dataset_name)) if matrix_to_plot_train[i, :].sum() > 0]
xticks_test  = [i for i in range(len(list_all_dataset_name)) if matrix_to_plot_test[i, :].sum() > 0]
xticks_labels_train = [f"{list_all_dataset_name[i][0:3]} ({i})" for i in xticks_train]
xticks_labels_test  = [f"{list_all_dataset_name[i][0:3]} ({i})" for i in xticks_test]

axs[0].set_xticks(xticks_train)
axs[0].set_xticklabels(xticks_labels_train, rotation = 90)
axs[1].set_xticks(xticks_test)
axs[1].set_xticklabels(xticks_labels_test, rotation = 90)

fig.tight_layout()
plt.show()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Save plot

if plot_config['n_samples_to_predict'] > 0 :
    folder_name = f"neurons_256_predict_samples_{plot_config['n_samples_to_predict']}"
else :
    folder_name = f"neurons_256_predict_portion_{int(plot_config['portion_of_signals_for_input'] * 100)}"

path_save_plot = f"saved_model/{folder_name}/0_comparison/"

if plot_config['normalize_0_1_range'] :
    path_save_plot += "normalized_"

if plot_config['clip_results'] :
    path_save_plot += "clipped_results_"

fig.savefig(path_save_plot + "comparison_average_error_prediction.png")






