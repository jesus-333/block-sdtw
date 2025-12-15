"""
Read the file obtained with comparison_avg_error_prediction_computation_V2.py and create a csv file.

Note that the data plotted here are the scores R defined as R = signal_length / length_dtw_path for each signal of each dataset.
This score has values between 0.5 (worst case, when length_dtw_path = 2 * signal_length) and 1 (best case, when length_dtw_path = signal_length).

@author: Alberto Zancanaro (Jesus)
@organization: Luxembourg Centre for Systems Biomedicine (LCSB)

Note to delete. The dataset PigAirwayPressure with OTW return some warnings during the computation of mean and std.
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

import numpy as np
import os
import pandas as pd
import pickle

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Settings

config = dict(
    # Info about the data used for training and prediction
    use_z_score_normalization = True,    # If True a z-score normalization will be applied signal by signal within each dataset
    portion_of_signals_for_input = 0.85, # Portion of the signals to use for training (the rest will be used for prediction)
    n_samples_to_predict = 100,            # Number of samples to predict (If negative it is ignored and the portion_of_signals_for_input is used to define the number of samples to predict. Otherwise, this parameter override portion_of_signals_for_input)
    epoch = -1,
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    rescale_score_in_0_1_range = False, # If True rescale the scores of each loss function in the range [0, 1] before plotting. The rescaling is done with the formula R_rescaled = 2 * R - 1, where R is the original score.
)

loss_function_to_plot = ['MSE', 'SDTW', 'SDTW_divergence', 'pruned_SDTW', 'OTW', 'block_SDTW_10', 'block_SDTW_50', 'block_SDTW_divergence_10', 'block_SDTW_divergence_50']
# loss_function_to_plot = ['MSE', 'SDTW', 'SDTW_divergence', 'pruned_SDTW', 'OTW', 'block_SDTW_10', 'block_SDTW_50']


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Load data

path_UCR_folder = "./data/UCRArchive_2018/"
list_all_dataset_name = [f for f in os.listdir(path_UCR_folder) if os.path.isdir(os.path.join(path_UCR_folder, f))]

if config['n_samples_to_predict'] > 0 :
    folder_name = f"neurons_256_predict_samples_{config['n_samples_to_predict']}"
else :
    folder_name = f"neurons_256_predict_portion_{int(config['portion_of_signals_for_input'] * 100)}"

if config['use_z_score_normalization'] : folder_name += '_z_score'

path_data = f"./saved_model/{folder_name}/0_comparison/"

if config['epoch'] == -1 : config['epoch'] = 'END'

with open(f"{path_data}score_lists_train_{config['epoch']}.pkl", "rb") as f : score_lists_train = pickle.load(f)
with open(f"{path_data}score_lists_test_{config['epoch']}.pkl", "rb") as f  : score_lists_test = pickle.load(f)

# Variable to save data
average_score_per_dataset_train = np.zeros((len(list_all_dataset_name), len(loss_function_to_plot)))
average_score_per_dataset_test  = np.zeros((len(list_all_dataset_name), len(loss_function_to_plot)))
std_score_per_dataset_train     = np.zeros((len(list_all_dataset_name), len(loss_function_to_plot)))
std_score_per_dataset_test      = np.zeros((len(list_all_dataset_name), len(loss_function_to_plot)))

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Compute average and std score per dataset
# config['dataset_to_plot'] = list(score_lists_train.keys())[0:3]

for i in range(len(list_all_dataset_name)) :
    # Get the name for the current dataset
    dataset_name = list_all_dataset_name[i]
    if dataset_name  == 'Missing_value_and_variable_length_datasets_adjusted' :
        continue

    # Get the score lists for the current dataset
    score_lists_dataset_train = score_lists_train[dataset_name]
    score_lists_dataset_test  = score_lists_test[dataset_name]

    # Skip the dataset if no score is available
    if len(score_lists_dataset_train) == 0 or len(score_lists_dataset_test) == 0 :
        continue
    
    # Compute the average and std score for each loss function
    for j in range(len(loss_function_to_plot)) :
        loss_function = loss_function_to_plot[j]

        average_score_per_dataset_train[i, j] = np.mean(score_lists_dataset_train[loss_function])
        average_score_per_dataset_test[i, j]  = np.mean(score_lists_dataset_test[loss_function])
        std_score_per_dataset_train[i, j]     = np.std(score_lists_dataset_train[loss_function])
        std_score_per_dataset_test[i, j]      = np.std(score_lists_dataset_test[loss_function])

    # Change NaN values to 0
    average_score_per_dataset_train[np.isnan(average_score_per_dataset_train)] = 0
    average_score_per_dataset_test[np.isnan(average_score_per_dataset_test)]   = 0
    std_score_per_dataset_train[np.isnan(std_score_per_dataset_train)]         = 0
    std_score_per_dataset_test[np.isnan(std_score_per_dataset_test)]           = 0

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Save data to csv file

# Create a DataFrame to save the data
columns = []
for loss_function in loss_function_to_plot :
    columns.append(f"{loss_function}_train_avg")
    columns.append(f"{loss_function}_train_std")
    columns.append(f"{loss_function}_test_avg")
    columns.append(f"{loss_function}_test_std")

data_to_save = pd.DataFrame(index = list_all_dataset_name, columns = columns)

for i in range(len(list_all_dataset_name)) :
    dataset_name = list_all_dataset_name[i]
    for j in range(len(loss_function_to_plot)) :
        loss_function = loss_function_to_plot[j]
        data_to_save.at[dataset_name, f"{loss_function}_train_avg"] = average_score_per_dataset_train[i, j]
        data_to_save.at[dataset_name, f"{loss_function}_train_std"] = std_score_per_dataset_train[i, j]
        data_to_save.at[dataset_name, f"{loss_function}_test_avg"]  = average_score_per_dataset_test[i, j]
        data_to_save.at[dataset_name, f"{loss_function}_test_std"]  = std_score_per_dataset_test[i, j]
        

# Save the DataFrame to a csv file

if config['n_samples_to_predict'] > 0 :
    folder_name = f"neurons_256_predict_samples_{config['n_samples_to_predict']}"
else :
    folder_name = f"neurons_256_predict_portion_{int(config['portion_of_signals_for_input'] * 100)}"

if config['use_z_score_normalization'] : folder_name += '_z_score'

path_csv_file = f"saved_model/{folder_name}/0_comparison/comparison_avg_error_prediction_{config['epoch']}.csv"
data_to_save.to_csv(path_csv_file)
print(f"Data saved to {path_csv_file}")






