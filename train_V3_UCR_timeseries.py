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
from torch import nn

import dataset
from model import MultiLayerPerceptron
from block_sdtw import reconstruction_loss

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Settings

bandwidth = 1
config = dict(
    # Training parameters
    portion_of_signals_for_input = 0.85, # Portion of the signals to use for training (the rest will be used for prediction)
    n_samples_to_predict = 100,            # Number of samples to predict (If negative it is ignored and the portion_of_signals_for_input is used to define the number of samples to predict. Otherwise, this parameter override portion_of_signals_for_input)
    batch_size = -1,                     # Batch size for training
    lr = 0.001,                          # Learning rate (lr)
    max_epochs = 60,                         # Number of epochs to train the model
    use_scheduler = True,                # Use the lr scheduler
    lr_decay_rate = 0.999,               # Parameter of the lr exponential scheduler
    optimizer_weight_decay = 1e-2,       # Weight decay of the optimizer
    alpha = 1,                           # Multiplier of the reconstruction error
    recon_loss_type = 1,                 # Loss function for the reconstruction (0 = L2, 1 = SDTW, 2 = SDTW-Divergence)
    block_size = 10,
    edge_samples_ignored = 0,            # Ignore this number of samples during the computation of the reconstructation loss
    gamma_dtw = 1,                       # Hyperparameter of the SDTW. Control the steepness of the soft-min inside the SDTW. The closer to 0 the closer the soft-min approximate the real min
    # device = "cuda" if torch.cuda.is_available() else "cpu",
    device = "mps",
    save_weights = True,                # Save the model weights after training. The path will be defined in the config['save_model_path'] + the name of the dataset
    save_model_path = "./saved_model/", # Path to save the model weights
    plot_and_save_prediction = True,     # Plot and save a prediction example after training
)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Dataset creation

path_UCR_folder = "./data/UCRArchive_2018/"

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
        print(f"Current Ratio: {config['block_size']/length_signal_to_predict:.3f} > 0.5")
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
    # Layers definition

    n_neurons = 256  # Number of neurons in the hidden layers

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
    # Model training

    if config['n_samples_to_predict'] > 0 :
        save_model_path_for_current_dataset = os.path.join(config['save_model_path'], f"block_size_{config['block_size']}_predict_samples_{config['n_samples_to_predict']}")
        save_model_path_for_current_dataset = os.path.join(save_model_path_for_current_dataset, name_dataset)
    else :
        save_model_path_for_current_dataset = os.path.join(config['save_model_path'], f"block_size_{config['block_size']}_predict_portion_{int(config['portion_of_signals_for_input']*100)}")
        save_model_path_for_current_dataset = os.path.join(save_model_path_for_current_dataset, name_dataset)

    print(f"Save model path: {save_model_path_for_current_dataset}")

    # Train and test MSE
    config['recon_loss_type'] = 0
    loss_function = reconstruction_loss(config)
    model = MultiLayerPerceptron(layers, loss_function, config)
    model.fit(x_1_train, x_2_train, config)
    y_pred_MSE = model(x_1_test).detach().cpu().numpy()
    if config['save_weights'] and not model.training_failed : model.save_model(save_model_path_for_current_dataset, filename = "model_MSE.pth")

    # Train and test SDTW
    config['recon_loss_type'] = 1
    loss_function = reconstruction_loss(config)
    model = MultiLayerPerceptron(layers, loss_function, config)
    model.fit(x_1_train, x_2_train, config)
    y_pred_SDTW = model(x_1_test).detach().cpu().numpy()
    if config['save_weights'] and not model.training_failed : model.save_model(save_model_path_for_current_dataset, filename = "model_SDTW.pth")

    # Train and test Pruned DTW
    config['recon_loss_type'] = 1
    config['bandwidth'] = bandwidth
    loss_function = reconstruction_loss(config)
    model = MultiLayerPerceptron(layers, loss_function, config)
    model.fit(x_1_train, x_2_train, config)
    y_pred_Pruned_DTW = model(x_1_test).detach().cpu().numpy()
    if config['save_weights'] and not model.training_failed : model.save_model(save_model_path_for_current_dataset, filename = "model_pruned_SDTW.pth")

    # Train and test block SDTW
    config['recon_loss_type'] = 3
    loss_function = reconstruction_loss(config)
    model = MultiLayerPerceptron(layers, loss_function, config)
    model.fit(x_1_train, x_2_train, config)
    y_pred_block_SDTW = model(x_1_test).detach().cpu().numpy()
    if config['save_weights'] and not model.training_failed : model.save_model(save_model_path_for_current_dataset, filename = "model_block_SDTW.pth")

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Plot results
    # Randomly select two time series from the test set and visualize the prediction
    if config['plot_and_save_prediction'] and not model.training_failed :

        if config['n_samples_to_predict'] > 0 :
            path_save_plot  = os.path.join(config['save_model_path'], f"block_size_{config['block_size']}_predict_samples_{config['n_samples_to_predict']}")
        else :
            path_save_plot = os.path.join(config['save_model_path'], f"block_size_{config['block_size']}_predict_portion_{int(config['portion_of_signals_for_input']*100)}")
        os.makedirs(path_save_plot, exist_ok = True)

        start_prediction = int(x_orig_test.shape[1] * config['portion_of_signals_for_input'])

        ts_index = np.random.randint(0, x_1_test.shape[0])
        tmp_path_save_plot = f"{path_save_plot}/0_plot_prediction/{name_dataset}_{ts_index}.png"
        dataset.visualize_prediction(ts_index, x_orig_test, start_prediction, y_pred_MSE, y_pred_SDTW, y_pred_Pruned_DTW, y_pred_block_SDTW,
                                     show_figure = False, path_save = tmp_path_save_plot, plot_title = f"Dataset: {name_dataset} - idx {ts_index}")

        ts_index = np.random.randint(0, x_1_test.shape[0])
        tmp_path_save_plot = f"{path_save_plot}/0_plot_prediction/{name_dataset}_{ts_index}.png"
        dataset.visualize_prediction(ts_index, x_orig_test, start_prediction, y_pred_MSE, y_pred_SDTW, y_pred_Pruned_DTW, y_pred_block_SDTW,
                                     show_figure = False, path_save = tmp_path_save_plot, plot_title = f"Dataset: {name_dataset} - idx {ts_index}")
