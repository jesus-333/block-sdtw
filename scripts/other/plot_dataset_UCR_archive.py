import matplotlib.pyplot as plt
import os

import dataset


# Get all the folders inside path_UCR_folder
path_UCR_folder = "./data/UCRArchive_2018/"
list_all_dataset_name = [f for f in os.listdir(path_UCR_folder) if os.path.isdir(os.path.join(path_UCR_folder, f))]
list_all_dataset_name = sorted(list_all_dataset_name, key=str.casefold)

idx_dataset_to_plot = 46
dataset_to_plot = list_all_dataset_name[idx_dataset_to_plot - 1]
path_save_plot = f"./Results/UCRARchive_samples_plot/{idx_dataset_to_plot}_{dataset_to_plot} (MALE)/"
samples_to_plot = 20

# Check if the dataset_to_plot is in the list of all dataset names
if dataset_to_plot not in list_all_dataset_name:
    raise ValueError(f"Dataset '{dataset_to_plot}' not found in the UCR Archive.")
else :
    print(f"Dataset '{dataset_to_plot}' found in the UCR Archive. Proceeding with plotting samples.")

# Path to the dataset folder
path_folder_dataset = os.path.join(path_UCR_folder, dataset_to_plot)

# Get training and test data
x_orig_train, _, x_orig_test, _ = dataset.read_UCR_dataset(path_folder_dataset, dataset_to_plot)

# Check if samples_to_plot is valid
if samples_to_plot <= 0 :
    samples_to_plot = min(len(x_orig_train), len(x_orig_test))
elif samples_to_plot > len(x_orig_train) or samples_to_plot > len(x_orig_test) :
    samples_to_plot = min(len(x_orig_train), len(x_orig_test))
    print(f"Warning: samples_to_plot is greater than the number of available samples. Setting samples_to_plot to {samples_to_plot}.")

# Create the directory if it doesn't exist
os.makedirs(path_save_plot, exist_ok=True)

# Plot the samples
for i in range(samples_to_plot) :
    print(f"Plotting sample {i+1} from {dataset_to_plot} dataset...")

    plt.figure(figsize = (16, 12))
    plt.plot(x_orig_train[i], label='Train Sample')
    # plt.plot(x_orig_test[i], label='Test Sample')
    plt.title(f'Sample {i+1} from {dataset_to_plot} Dataset')
    plt.xlabel('Time Steps')
    plt.ylabel('Value')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    
    
    # Save the plot
    plt.savefig(os.path.join(path_save_plot, f'sample_{i+1}.png'))
    plt.close()

