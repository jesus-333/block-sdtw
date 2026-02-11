"""
Create a CSV file with the name of the datasets inside the UCR Archive.
For each dataset insert also the number of samples per signals and if it is included or excluded during the training.

@author: Alberto Zancanaro (Jesus)
@organization: Luxembourg Centre for Systems Biomedicine (LCSB)
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

import os
import pandas as pd

import dataset

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Settings

n_samples_to_predict = 100

path_UCR_folder = "./data/UCRArchive_2018/"
path_save_csv = "./other_analysis/ucr_datasets_description.csv"

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# Get all the folders inside path_UCR_folder
list_all_dataset_name = [f for f in os.listdir(path_UCR_folder) if os.path.isdir(os.path.join(path_UCR_folder, f))]

list_n_samples_per_signal = []
list_portion_of_signal_to_predict = []
list_datasets_skipped = []

for i in range(len(list_all_dataset_name)):
    name_dataset = list_all_dataset_name[i]

    # This folder has a different structure respect the other UCR dataset so I will skip it for now
    if name_dataset == 'Missing_value_and_variable_length_datasets_adjusted' :
        list_n_samples_per_signal.append(-1)
        list_portion_of_signal_to_predict.append(-1)
        list_datasets_skipped.append("Different folder structure")
        continue

    print("Dataset {}: {}".format(i, name_dataset))
    
    # Path to the dataset folder
    path_folder_dataset = os.path.join(path_UCR_folder, name_dataset)

    # Get training and test data
    x_orig_train, _, x_orig_test, _ = dataset.read_UCR_dataset(path_folder_dataset, name_dataset)

    n_samples_signal = x_orig_train.shape[1]
    list_n_samples_per_signal.append(n_samples_signal)

    # Check portion_of_signals_for_input
    if n_samples_to_predict > 0 :
        portion_of_signals_for_input = (n_samples_signal - n_samples_to_predict) / n_samples_signal

        if n_samples_to_predict >= n_samples_signal :
            print(f"Skipping dataset {name_dataset} because the number of samples to predict ({n_samples_to_predict}) is greater than or equal to the length of the signals ({x_orig_train.shape[1]})")
            list_portion_of_signal_to_predict.append(-1)
            list_datasets_skipped.append("Signals too short")
        elif n_samples_to_predict / n_samples_signal >= 0.5 :
            print(f"Skipping dataset {name_dataset} because the number of samples to predict ({n_samples_to_predict}) is too high compared to the length of the signals ({x_orig_train.shape[1]})")
            print("The number of samples to predict should be less than half of the length of the signals (i.e. n_samples_to_predict/length_signal < 0.5)")
            print(f"Current Ratio: {n_samples_to_predict / n_samples_signal:.3f} >= 0.5")
            list_portion_of_signal_to_predict.append(n_samples_to_predict / n_samples_signal)
            list_datasets_skipped.append("Portion to predict too short")
        else :
            print(f"Dataset {name_dataset} used for the training")
            list_portion_of_signal_to_predict.append(n_samples_to_predict / n_samples_signal)
            list_datasets_skipped.append('-')

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Convert to DataFrame and save as CSV

# Create DataFrame
pandas_dict = {
    'Dataset_Name' : list_all_dataset_name,
    'N_Samples_Per_Signal' : list_n_samples_per_signal,
    'Portion_Of_Signal_To_Predict' : list_portion_of_signal_to_predict,
    'Reason_For_Skipping' : list_datasets_skipped
}
df_datasets_description = pd.DataFrame(pandas_dict)

# Save DataFrame as CSV
os.makedirs(os.path.dirname(path_save_csv), exist_ok = True)
df_datasets_description.to_csv(path_save_csv, index = False)
print(f"CSV file saved at: {path_save_csv}")
