"""
Plot the results obtained with comparison_avg_error_prediction_computation_V2.py
The main difference with respect to comparison_avg_error_prediction_plot_V2.py is that here we plot the histogram of all the scores obtained instead of the average score per dataset.

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
    portion_of_signals_for_input = 0.85, # Portion of the signals to use for training (the rest will be used for prediction)
    n_samples_to_predict = -1,            # Number of samples to predict (If negative it is ignored and the portion_of_signals_for_input is used to define the number of samples to predict. Otherwise, this parameter override portion_of_signals_for_input)
    n_bins = 50,                           # Number of bins to use for the histogram
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    figsize = (18, 12),
    cmap = 'Reds',
    # cmap = 'RdYlGn',
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

# Load matrix with the average errors for all loss functions
all_scores_train = np.load(path_errors_matrix_folder + "score_matrix_train.npy")
all_scores_test  = np.load(path_errors_matrix_folder + "score_matrix_test.npy")

# Create matrix to store the average errors for the selected loss functions
matrix_to_plot_train = np.zeros((len(list_all_dataset_name), len(loss_function_to_plot)))
matrix_to_plot_test = np.zeros((len(list_all_dataset_name), len(loss_function_to_plot)))

# Fill the matrix to plot
for i in range(len(loss_function_to_plot)) :
    idx = loss_function_to_idx[loss_function_to_plot[i]]
    matrix_to_plot_train[:, i] = matrix_with_all_data_train[:, idx]
    matrix_to_plot_test[:, i]  = matrix_with_all_data_test[:, idx]

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
    axs[i].set_xticks(np.arange(len(list_all_dataset_name)) + 0.5, minor = True,)
    # axs[i].set_yticks(np.arange(3) + 0.5, minor = True)
    axs[i].set_xticklabels([], minor = True)
    axs[i].grid(which = 'minor', color = 'w', linestyle = '-', snap = False)
    # axs[i].set_xticklabels(list_all_dataset_name, rotation = 90)

    axs[i].set_yticks(np.arange(len(loss_function_to_plot)), minor = False)
    axs[i].set_yticklabels(loss_function_to_plot)
    axs[i].set_title('Train set' if i == 0 else 'Test set')

    axs[i].set_xlabel('Datasets')

    # Add colorbar
    cbar = fig.colorbar(axs[i].images[0], ax = axs[i], fraction = 0.046, pad = 0.04)
    

# Set x-ticks only for datasets that were used (i.e., rows that have at least one non-zero value)
xticks_train = [i for i in range(len(list_all_dataset_name)) if matrix_to_plot_train[i, :].sum() > 0]
xticks_test  = [i for i in range(len(list_all_dataset_name)) if matrix_to_plot_test[i, :].sum() > 0]
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
fig.suptitle(overall_title, fontsize = 16)

# Adjust layout and show plot
fig.tight_layout()
plt.show()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Save plot

if plot_config['n_samples_to_predict'] > 0 :
    folder_name = f"neurons_256_predict_samples_{plot_config['n_samples_to_predict']}"
else :
    folder_name = f"neurons_256_predict_portion_{int(plot_config['portion_of_signals_for_input'] * 100)}"

path_save_plot = f"saved_model/{folder_name}/0_comparison/"

fig.savefig(path_save_plot + "comparison_average_score_prediction.png")






