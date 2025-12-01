"""
This script is very similar to train_V2_UCR_timeseries.py but it will execute the training on all the UCR datasets
The main difference are
- Additional check before training to avoid training when certain conditions are not respected (e.g. portion of the signal to predict is too small)
- Minor modification for saved model names/paths

Note : the UCR archive was downloaded from the link below and unzipped in a folder named 'data'
Link download : https://www.cs.ucr.edu/%7Eeamonn/time_series_data_2018/

@author: Alberto Zancanaro (Jesus)
@organization: Luxembourg Centre for Systems Biomedicine (LCSB)
"""
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

import numpy as np
import os
import torch
from torch import nn

import dataset
from model import MultiLayerPerceptron
from block_sdtw import reconstruction_loss

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Settings

bandwidth = 1
config = dict(
    # Training parameters
    use_z_score_normalization = True,    # If True a z-score normalization will be applied signal by signal within each dataset
    portion_of_signals_for_input = 0.85, # Portion of the signals to use for training (the rest will be used for prediction)
    n_samples_to_predict = -1,            # Number of samples to predict (If negative it is ignored and the portion_of_signals_for_input is used to define the number of samples to predict. Otherwise, this parameter override portion_of_signals_for_input)
    batch_size = -1,                     # Batch size for training
    lr = 0.001,                          # Learning rate (lr)
    max_epochs = 60,                         # Number of epochs to train the model
    use_scheduler = True,                # Use the lr scheduler
    lr_decay_rate = 0.999,               # Parameter of the lr exponential scheduler
    optimizer_weight_decay = 1e-2,       # Weight decay of the optimizer
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # General loss config
    recon_loss_type = 1,                 # Loss function for the reconstruction (0 = L2, 1 = SDTW, 2 = SDTW-Divergence)
    alpha = 1,                           # Multiplier of the reconstruction error
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Soft-DTW/block SDTW config
    block_size = 10,
    edge_samples_ignored = 0,            # Ignore this number of samples during the computation of the reconstructation loss
    gamma_dtw = 1,                       # Hyperparameter of the SDTW. Control the steepness of the soft-min inside the SDTW. The closer to 0 the closer the soft-min approximate the real min
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # OTW config
    s = 0.5,
    beta = 1,
    m = 1,
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    device = "cuda" if torch.cuda.is_available() else "cpu",
    # device = "mps",
    save_weights = True,                # Save the model weights after training. The path will be defined in the config['save_model_path'] + the name of the dataset
    save_model_path = "./saved_model/", # Path to save the model weights
    save_every_n_epoch = 10,             # If positive save the model every n epochs
    plot_and_save_prediction = True,     # Plot and save a prediction example after training
)

loss_function_to_use_list = ['MSE', 'SDTW', 'SDTW_divergence', 'pruned_SDTW', 'OTW', 'block_SDTW']
loss_function_to_use_list = ['block_SDTW']

n_neurons = 256  # Number of neurons in the hidden layers

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Dataset creation

path_UCR_folder = "./data/UCRArchive_2018/"

loss_function_to_recon_loss_type = dict(
    MSE = 0,
    SDTW = 1,
    SDTW_divergence = 2,
    pruned_SDTW = 1,
    OTW = 5,
    block_SDTW = 3,
)

# Get all the folders inside path_UCR_folder
list_all_dataset_name = [f for f in os.listdir(path_UCR_folder) if os.path.isdir(os.path.join(path_UCR_folder, f))]

for i in range(len(list_all_dataset_name)):
    name_dataset = list_all_dataset_name[i]

    # This folder has a different structure respect the other UCR dataset so I will skip it for now
    if name_dataset == 'Missing_value_and_variable_length_datasets_adjusted' : continue

    print("Dataset {}: {}".format(i, name_dataset))
    
    # Path to the dataset folder
    path_folder_dataset = os.path.join(path_UCR_folder, name_dataset)

    # Get training and test data
    x_orig_train, _, x_orig_test, _ = dataset.read_UCR_dataset(path_folder_dataset, name_dataset)

    # Check portion_of_signals_for_input
    if config['n_samples_to_predict'] > 0 :
        config['portion_of_signals_for_input'] = (x_orig_train.shape[1] - config['n_samples_to_predict']) / x_orig_train.shape[1]

        if config['n_samples_to_predict'] >= x_orig_train.shape[1] :
            print(f"Skipping dataset {name_dataset} because the number of samples to predict ({config['n_samples_to_predict']}) is greater than or equal to the length of the signals ({x_orig_train.shape[1]})")
            continue
        
        if config['n_samples_to_predict'] / x_orig_train.shape[1] >= 0.5 :
            print(f"Skipping dataset {name_dataset} because the number of samples to predict ({config['n_samples_to_predict']}) is too high compared to the length of the signals ({x_orig_train.shape[1]})")
            print("The number of samples to predict should be less than half of the length of the signals (i.e. n_samples_to_predict/length_signal < 0.5)")
            print(f"Current Ratio: {config['n_samples_to_predict']/x_orig_train.shape[1]:.3f} >= 0.5")
            continue

    # (OPTIONAL) Z-score normalization
    if config['use_z_score_normalization'] :
        x_orig_train = dataset.z_score_normalization(x_orig_train)
        x_orig_test = dataset.z_score_normalization(x_orig_test)

    # Divide the training data in input signal and signal to predict (TRAIN)
    x_1_train, x_2_train = dataset.generate_signals_V2(x_orig_train, int(x_orig_train.shape[1] * config['portion_of_signals_for_input']))

    # Divide the test data in input signal and signal to predict (TEST)
    x_1_test, x_2_test = dataset.generate_signals_V2(x_orig_test, int(x_orig_test.shape[1] * config['portion_of_signals_for_input']))

    print("Shape of x_1_train: ", x_1_train.shape)
    print("Shape of x_2_train: ", x_2_train.shape)
    print("Shape of x_1_test: ", x_1_test.shape)
    print("Shape of x_2_test: ", x_2_test.shape)

    length_signal_to_predict = x_2_train.shape[1]

    if length_signal_to_predict <= config['block_size'] :
        print(f"Skipping dataset {name_dataset} because the portion of the signal to predict ({length_signal_to_predict}) is too small compared to the block size ({config['block_size']})")
        print("fThe length of the signal to predict should be at least twice the block size (i.e. block_size/length_signal_to_predict <= 0.5)")
        print(f"Current Ratio: {config['block_size'] / length_signal_to_predict:.3f} > 0.5")
        continue

    # Visualize randomly 4 pairs of signals
    # for i in range(4) :
    #     idx = np.random.randint(0, n_signals_to_generate)
    #     dataset.visualize_signals(x_1_train[idx], x_2_train[idx], x_orig_train[idx], t_orig_train[idx])

    # Adjust the shape of the training and test data for the time series format
    # This is a workaround to ensure compatibility with the Soft-DTW loss function implementation
    x_1_train = np.expand_dims(x_1_train, axis = -1)
    x_2_train = np.expand_dims(x_2_train, axis = -1)
    x_1_test = np.expand_dims(x_1_test, axis = -1)
    x_2_test = np.expand_dims(x_2_test, axis = -1)

    if x_1_train.shape[0] >= 100 : config['batch_size'] = 100

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Model training
    
    # Define save model path
    if config['n_samples_to_predict'] > 0 :
        folder_name = f"neurons_{n_neurons}_predict_samples_{config['n_samples_to_predict']}"
    else :
        folder_name = f"neurons_{n_neurons}_predict_portion_{int(config['portion_of_signals_for_input'] * 100)}"

    if config['use_z_score_normalization'] : folder_name += '_z_score'

    save_model_path_for_current_dataset = os.path.join(config['save_model_path'], folder_name)
    save_model_path_for_current_dataset = os.path.join(save_model_path_for_current_dataset, name_dataset)

    print(f"Save model path: {save_model_path_for_current_dataset}")

    # Dictionary to store prediction results
    prediction_results = dict()

    for i in range(len(loss_function_to_use_list)) :
        # Get loss function
        loss_function_to_use = loss_function_to_use_list[i]
        config['recon_loss_type'] = loss_function_to_recon_loss_type[loss_function_to_use]
        loss_function = reconstruction_loss(config)

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Layers definition

        # V1 (1 hidden layer)
        layers = nn.Sequential(
            nn.Linear(in_features = x_1_train.shape[1], out_features = n_neurons),
            nn.GELU(),
            nn.Linear(in_features = n_neurons, out_features = x_2_train.shape[1])
        )

        # V2 (2 hidden layers)
        layers = nn.Sequential(
            nn.Linear(in_features = x_1_train.shape[1], out_features = n_neurons),
            nn.GELU(),
            nn.Linear(in_features = n_neurons, out_features = n_neurons),
            nn.GELU(),
            nn.Linear(in_features = n_neurons, out_features = x_2_train.shape[1])
        )

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Create model

        model = MultiLayerPerceptron(layers, loss_function, config)
        model.save_model_path = save_model_path_for_current_dataset

        model_name = f'model_block_SDTW_{config["block_size"]}' if loss_function_to_use == 'block_SDTW' else f'model_{loss_function_to_use}'
        model.model_name = f"{model_name}"

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Train model
        model.fit(x_1_train, x_2_train, config)

        # Save prediction on test set and model weights
        if not model.training_failed :
            prediction_results[loss_function_to_use] = model(x_1_test).detach().cpu().numpy()
            if config['save_weights'] : model.save_model(save_model_path_for_current_dataset, filename = f"{model_name}_END")

        del model, layers, loss_function

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Plot results
    # Randomly select two time series from the test set and visualize the prediction
    if config['plot_and_save_prediction'] and len(prediction_results) > 0 :

        if config['n_samples_to_predict'] > 0 :
            folder_name = f"neurons_{n_neurons}_predict_samples_{config['n_samples_to_predict']}"
        else :
            folder_name = f"neurons_{n_neurons}_predict_portion_{int(config['portion_of_signals_for_input'] * 100)}"

        if config['use_z_score_normalization'] : folder_name += '_z_score'

        path_save_plot = os.path.join(config['save_model_path'], folder_name)
        os.makedirs(path_save_plot, exist_ok = True)

        start_prediction = int(x_orig_test.shape[1] * config['portion_of_signals_for_input'])

        x_pred_list = []
        x_pred_label = []

        for loss_function_to_use in prediction_results :
            x_pred_list.append(prediction_results[loss_function_to_use])
            x_pred_label.append(loss_function_to_use)

        ts_index = np.random.randint(0, x_1_test.shape[0])
        tmp_path_save_plot = f"{path_save_plot}/0_plot_prediction/{name_dataset}_{ts_index}.png"
        dataset.visualize_prediction(ts_index, x_orig_test, start_prediction, x_pred_list, x_pred_label,
                                     show_figure = False, path_save = tmp_path_save_plot, plot_title = f"Dataset: {name_dataset} - idx {ts_index}")

        ts_index = np.random.randint(0, x_1_test.shape[0])
        tmp_path_save_plot = f"{path_save_plot}/0_plot_prediction/{name_dataset}_{ts_index}.png"
        dataset.visualize_prediction(ts_index, x_orig_test, start_prediction, x_pred_list, x_pred_label,
                                     show_figure = False, path_save = tmp_path_save_plot, plot_title = f"Dataset: {name_dataset} - idx {ts_index}")
