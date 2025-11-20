"""
Compare the prediction of model trained with the various loss functions (MSE, SDTW, SDTW divergence, Pruned SDTW, OTW, Block SDTW)
For each dataset the average prediction error is computed.

To compute the average prediction error we follow these steps:
- Train the model with the various loss functions (script train_V3_UCR_timeseries.py) (The training is done with raw data)
- Load the specific dataset and load the weights of the model trained on that dataset
- Compute the signal prediction on the test set (for each model obtained with the various loss functions)
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
import numpy as np
import os
import torch

import dataset
from model import MultiLayerPerceptron

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Settings

model_config = dict(
    # Training parameters
    portion_of_signals_for_input = 0.85, # Portion of the signals to use for training (the rest will be used for prediction)
    n_samples_to_predict = -1,            # Number of samples to predict (If negative it is ignored and the portion_of_signals_for_input is used to define the number of samples to predict. Otherwise, this parameter override portion_of_signals_for_input)
    epoch = -1,
    block_size = 50,
    device = "mps",
    model_weights_path = "./saved_model/", # Path to save the model weights
    normalize_0_1_range = True,        # Normalize each signal between 0 and 1 before DTW computation
)

loss_function_to_use_list = ['MSE', 'SDTW', 'SDTW_divergence', 'pruned_SDTW', 'OTW', 'block_SDTW_10', 'block_SDTW_50']

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
# Get dataset names

path_UCR_folder = "./data/UCRArchive_2018/"

# Get all the folders inside path_UCR_folder
list_all_dataset_name = [f for f in os.listdir(path_UCR_folder) if os.path.isdir(os.path.join(path_UCR_folder, f))]

# Matrix to store the average errors
# TODO add the possibility to load matrices from previous computations and update them
average_errors_matrix_train = np.zeros((len(list_all_dataset_name), 7)) # Columns: SDTW, Pruned SDTW, Block SDTW
average_errors_matrix_test  = np.zeros((len(list_all_dataset_name), 7)) # Columns: SDTW, Pruned SDTW, Block SDTW

# variable to store distance lists (they will be later converted to numpy arrays)
distance_matrix_train = []
distance_matrix_test  = []

original_n_samples_to_predict = model_config['n_samples_to_predict']

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def get_model_and_load_weights(model_config : dict, model_weights_path : str, training_modality : str, epoch : int = -1) :
    """
    Get the model and load the weights from the specified path.

    Parameters
    ----------
    model_config : dict
        Dictionary containing the model configuration.
    model_weights_path : str
        Path to the model weights.
    training_modality : str
        Training modality. Used to load the correct model weights.
    """

    # Number of neurons in the hidden layers
    n_neurons = 256

    # V2 (2 hidden layers)
    layers = torch.nn.Sequential(
        torch.nn.Linear(in_features = x_1_train.shape[1], out_features = n_neurons),
        torch.nn.GELU(),
        torch.nn.Linear(in_features = n_neurons, out_features = n_neurons),
        torch.nn.GELU(),
        torch.nn.Linear(in_features = n_neurons, out_features = model_config['n_samples_to_predict'])
    )
    
    # Create model
    model = MultiLayerPerceptron(layers, loss_function = None, config = model_config)

    # Load model weights
    if epoch == -1 : epoch = 'END'
    model_weights_path = f"{model_weights_path}model_{training_modality}_{epoch}.pth"
    model.load_state_dict(torch.load(model_weights_path, map_location = model_config['device']))

    return model

def compute_average_error(x_input, x_ground_truth, model, model_config : dict) :
    """
    Compute the average error between the predicted signal and the ground truth signal using standard DTW.
    The predicted signal is obtained by passing x_input through the model.
    
    Parameters
    ----------
    x_input : np.ndarray
        Input signal to the model. It has shape (n_signals, input_signal_length)
    x_ground_truth : np.ndarray
        Ground truth signal. It has shape (n_signals, output_signal_length)
    model : torch.nn.Module
        Trained model to use for prediction.
    model_config : dict
        Dictionary containing the model configuration.
    """

    # Get model predictions
    x_pred = predict_signal(x_input, model, model_config)

    if model_config['normalize_0_1_range'] :
        # Get min value for the entire signal
        min_input = np.min(x_input, axis = 1, keepdims = True)
        min_ground_truth = np.min(x_ground_truth, axis = 1, keepdims = True)
        min_signal = np.minimum(min_input, min_ground_truth)

        # Get max value for the entire signal
        max_input = np.max(x_input, axis = 1, keepdims = True)
        max_ground_truth = np.max(x_ground_truth, axis = 1, keepdims = True)
        max_signal = np.maximum(max_input, max_ground_truth)

        # Note that the original signal is the concatenation of x_input and x_ground_truth
        # It is split in two parts for training and prediction purposes
        # When we normalize between 0 and 1 we want to do that coherently for both parts
        # So we use the min and max values computed on the entire signal (input + ground)

        # Normalize ground truth signal between 0 and 1
        x_ground_truth = (x_ground_truth - min_signal) / (max_signal - min_signal)

        # Normalize predicted signal between 0 and 1
        x_pred = (x_pred - min_signal) / (max_signal - min_signal)

    distance_list = compute_distance_ground_truth_prediction(x_ground_truth, x_pred)

    # Compute average error over all samples
    average_error = np.mean(distance_list)

    return average_error, distance_list

def predict_signal(x_input, model, model_config) :
    """
    Parameters
    ----------
    x_input : np.ndarray
        Input signal to the model. It has shape (n_signals, input_signal_length)
    model : torch.nn.Module
        Trained model to use for prediction.
    model_config : dict
        Dictionary containing the model configuration.
    """
    
    # Set the model to evaluation mode and move it to the correct device
    model.eval()
    model.to(model_config['device'])

    # Convert input to torch tensor
    x_input_tensor = torch.from_numpy(x_input).float().to(model_config['device'])

    # Get model predictions
    x_pred_tensor = model(x_input_tensor).detach().cpu().squeeze().numpy()

    return x_pred_tensor

def compute_distance_ground_truth_prediction(x_ground_truth, x_predict) -> list :
    """
    Compute the average error between the predicted signal and the ground truth signal using standard DTW.
    The predicted signal is obtained by passing x_input through the model.
    
    Parameters
    ----------
    x_ground_truth : np.ndarray
        Ground truth signal. It has shape (n_signals, output_signal_length)
    x_predict : np.ndarray
        Predicted signal. It has shape (n_signals, output_signal_length)

    Returns
    -------
    distance_list : list
        List containing the average error per signal in the batch.
    """

    distance_list = []

    for i in range(x_ground_truth.shape[0]) :
        x_gt_signal = x_ground_truth[i, :]
        x_pred_signal = x_predict[i, :]

        # Compute DTW distance
        dtw_distance = dtw.dtw(x_gt_signal, x_pred_signal).distance

        # Average error per sample
        average_error = dtw_distance / x_ground_truth.shape[1]

        distance_list.append(average_error)

    return distance_list

max_n_signals = -1

for i in range(len(list_all_dataset_name)):
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Check if the trained model exists for the current dataset/settings

    # Get dataset name
    name_dataset = list_all_dataset_name[i]

    # This folder has a different structure respect the other UCR dataset so the trained was not performed
    if name_dataset == 'Missing_value_and_variable_length_datasets_adjusted' : continue
    
    # Path to the model weights
    if model_config['n_samples_to_predict'] > 0 :
        folder_name = f"neurons_256_predict_samples_{model_config['n_samples_to_predict']}"
    else :
        folder_name = f"neurons_256_predict_portion_{int(model_config['portion_of_signals_for_input'] * 100)}"

    path_weights = f"{model_config['model_weights_path']}{folder_name}/{name_dataset}/"
    print(path_weights)

    # Check if the model weights folder exists
    if not os.path.exists(path_weights) :
        print(f"Dataset {i}: {name_dataset} - No trained model found, skipping... ({round((i + 1) / len(list_all_dataset_name) * 100, 2)}%)")
        continue
    else :
        print(f"Dataset {i}: {name_dataset} ({round((i + 1) / len(list_all_dataset_name) * 100, 2)}%)")
    
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Load dataset

    # Path to the dataset folder
    path_folder_dataset = os.path.join(path_UCR_folder, name_dataset)

    # Get training and test data
    x_orig_train, _, x_orig_test, _ = dataset.read_UCR_dataset(path_folder_dataset, name_dataset)

    # Update portion_of_signals_for_input if n_samples_to_predict is set. portion_of_signals_for_input is used to divide the signals in input signal and signal to predict
    # Remember that n_samples_to_predict override portion_of_signals_for_input if it is > 0
    if model_config['n_samples_to_predict'] > 0 : model_config['portion_of_signals_for_input'] = (x_orig_train.shape[1] - model_config['n_samples_to_predict']) / x_orig_train.shape[1]

    # Divide the training data in input signal and signal to predict (TRAIN)
    x_1_train, x_2_train = dataset.generate_signals_V2(x_orig_train, int(x_orig_train.shape[1] * model_config['portion_of_signals_for_input']))

    # Divide the test data in input signal and signal to predict (TEST)
    x_1_test, x_2_test = dataset.generate_signals_V2(x_orig_test, int(x_orig_test.shape[1] * model_config['portion_of_signals_for_input']))
    
    # Update n_samples_to_predict (it is used in the model definition)
    # Note that this happens only if n_samples_to_predict < 0
    if model_config['n_samples_to_predict'] < 0 : model_config['n_samples_to_predict'] = x_2_train.shape[1]

    for j in range(len(loss_function_to_use_list)) :
        # Get loss functions label and index
        loss_function_to_use = loss_function_to_use_list[j]
        idx_loss_function = loss_function_to_idx[loss_function_to_use]
        print(f"  - Loss function: {loss_function_to_use} ({j + 1}/{len(loss_function_to_use_list)})")

        # Get model and load weights
        try :
            model_for_prediction = get_model_and_load_weights(model_config, path_weights, loss_function_to_use, model_config['epoch'])
        except FileNotFoundError :
            print(f"    Warning: Model weights not found for loss function {loss_function_to_use}, skipping...")
            average_errors_matrix_train[i, idx_loss_function] = 0
            average_errors_matrix_test[i, idx_loss_function]  = 0
            distance_matrix_train.append([])
            distance_matrix_test.append([])
            continue

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Compute model predictions and compute average error (TRAIN)
        try :
            average_error_train, distance_list_train = compute_average_error(x_1_train, x_2_train, model_for_prediction, model_config)
        except ValueError as e :
            average_error_train, distance_list_train = 0, []
            print(f"Warning: {e}")
        average_errors_matrix_train[i, idx_loss_function] = average_error_train
        distance_matrix_train.append(distance_list_train)

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Compute model predictions and compute average error (TEST)
        try :
            average_error_test, distance_list_test = compute_average_error(x_1_test, x_2_test, model_for_prediction, model_config)
        except ValueError as e:
            average_error_test, distance_list_test = 0, []
            print(f"Warning: {e}")
        average_errors_matrix_test[i, idx_loss_function] = average_error_test
        distance_matrix_test.append(distance_list_test)

        # Update max_n_signals
        if max_n_signals < len(distance_list_train) : max_n_signals = len(distance_list_train)
        if max_n_signals < len(distance_list_test)  : max_n_signals = len(distance_list_test)
        
    # Restore n_samples_to_predict in the model config
    # This is needed when we compute the errors for the models trained with portion_of_signals_for_input
    # Due to computation reasons we still use n_samples_to_predict in the dataset generation
    # So even if we originally set n_samples_to_predict < 0, it will be later set to a positive value corresponding to (1 - portion_of_signals_for_input) * signal_length
    # To avoid error for those cases we restore the original value of n_samples_to_predict
    # In case we had set n_samples_to_predict > 0 from the beginning this will not change anything
    model_config['n_samples_to_predict'] = original_n_samples_to_predict

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Convert distance lists to numpy arrays

tmp_distance_matrix_train = np.zeros((len(distance_matrix_train), max_n_signals))
tmp_distance_matrix_test  = np.zeros((len(distance_matrix_test), max_n_signals))

for i in range(len(distance_matrix_train)) :
    distance_list_train = distance_matrix_train[i]
    distance_list_test  = distance_matrix_test[i]

    tmp_distance_matrix_train[i, 0:len(distance_list_train)] = np.array(distance_list_train)
    tmp_distance_matrix_test[i, 0:len(distance_list_test)]   = np.array(distance_list_test)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Save average errors matrices

if model_config['n_samples_to_predict'] > 0 :
    folder_name = f"neurons_256_predict_samples_{model_config['n_samples_to_predict']}"
else :
    folder_name = f"neurons_256_predict_portion_{int(model_config['portion_of_signals_for_input'] * 100)}"

path_save = f"{model_config['model_weights_path']}{folder_name}/0_comparison/"

# Create folder if it does not exist
if not os.path.exists(path_save) : os.makedirs(path_save)

# Save average errors
if model_config['normalize_0_1_range'] :
    # Save normalized average errors and distance matrices as numpy arrays
    np.save(f"{path_save}normalized_average_errors_matrix_train.npy", average_errors_matrix_train)
    np.save(f"{path_save}normalized_average_errors_matrix_test.npy", average_errors_matrix_test)
    np.save(f"{path_save}normalized_distance_matrix_train.npy", tmp_distance_matrix_train)
    np.save(f"{path_save}normalized_distance_matrix_test.npy", tmp_distance_matrix_test)
else :
    # Save average errors and distance matrices as numpy arrays
    np.save(f"{path_save}average_errors_matrix_train.npy", average_errors_matrix_train)
    np.save(f"{path_save}average_errors_matrix_test.npy", average_errors_matrix_test)
    np.save(f"{path_save}distance_matrix_train.npy", tmp_distance_matrix_train)
    np.save(f"{path_save}distance_matrix_test.npy", tmp_distance_matrix_test)











