"""
Variation of the V2 script to work with the UCR Time Series Classification Archive data

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
import otw

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Settings

bandwidth = 1
config = dict(
    # Training parameters
    n_dataset_to_use = 15,               # Number of datasets to use from the UCR archive (See notes below for more details)
    portion_of_signals_for_input = -1,  # Portion of the signals to use for training (the rest will be used for prediction)(If negative, the last 100 samples will be used for prediction if the signal is longer than 100 samples; otherwise, the last 50% of the signal will be used for prediction)
    batch_size = -1,                     # Batch size for training
    lr = 0.001,                          # Learning rate (lr)
    max_epochs = 20,                         # Number of epochs to train the model
    use_scheduler = True,                # Use the lr scheduler
    lr_decay_rate = 0.999,               # Parameter of the lr exponential scheduler
    optimizer_weight_decay = 1e-2,       # Weight decay of the optimizer
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # General loss parameters
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
    # Technical stuff
    # device = "cuda" if torch.cuda.is_available() else "cpu",
    device = "mps",
    save_weights = True,                # Save the model weights after training. The path will be defined in the config['save_model_path'] + the name of the dataset
    save_model_path = "./saved_models/", # Path to save the model weights
)


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Dataset creation

path_UCR_folder = "./data/UCRArchive_2018/"

# Get all the folders inside path_UCR_folder
list_all_dataset_name = [f for f in os.listdir(path_UCR_folder) if os.path.isdir(os.path.join(path_UCR_folder, f))]

# Sample n dataset to use from the list of all dataset names
# Since the UCR archive contains a lot of datasets (129 at the moment of writing), we sample a subset of them to use in the training
if config['n_dataset_to_use'] > 0 and config['n_dataset_to_use'] < len(list_all_dataset_name) :
    list_dataset_to_use = np.random.choice(list_all_dataset_name, size = config['n_dataset_to_use'], replace = False)
else :
    list_dataset_to_use = list_all_dataset_name

for i in range(len(list_dataset_to_use)):
    name_dataset = list_dataset_to_use[i]

    # This folder has a different structure respect the other UCR dataset so I will skip it for now
    if name_dataset == 'Missing_value_and_variable_length_datasets_adjusted' : continue

    print("Dataset {}: {}".format(i, name_dataset))
    
    # Path to the dataset folder
    path_folder_dataset = os.path.join(path_UCR_folder, name_dataset)

    # Get training and test data
    x_orig_train, _, x_orig_test, _ = dataset.read_UCR_dataset(path_folder_dataset, name_dataset)

    # Check portion_of_signals_for_input
    if config['portion_of_signals_for_input'] < 0 :
        # If negative, use the last 100 samples for prediction if the signal is longer than 100 samples; otherwise, use the last 50% of the signal for prediction
        if x_orig_train.shape[1] > 100 :
            config['portion_of_signals_for_input'] = (x_orig_train.shape[1] - 100) / x_orig_train.shape[1]
        else :
            config['portion_of_signals_for_input'] = 0.5

    # Divide the training data in input signal and signal to predict (TRAIN)
    x_1_train, x_2_train = dataset.generate_signals_V2(x_orig_train, int(x_orig_train.shape[1] * config['portion_of_signals_for_input']))

    # Divide the test data in input signal and signal to predict (TEST)
    x_1_test, x_2_test = dataset.generate_signals_V2(x_orig_test, int(x_orig_test.shape[1] * config['portion_of_signals_for_input']))

    print("Shape of x_1_train: ", x_1_train.shape)
    print("Shape of x_2_train: ", x_2_train.shape)
    print("Shape of x_1_test: ", x_1_test.shape)
    print("Shape of x_2_test: ", x_2_test.shape)

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

    save_model_path_for_current_dataset = os.path.join(config['save_model_path'], name_dataset)

    # Train and test MSE
    config['recon_loss_type'] = 0
    loss_function = reconstruction_loss(config)
    model = MultiLayerPerceptron(layers, loss_function, config)
    model.fit(x_1_train, x_2_train, config)
    x_pred_MSE = model(x_1_test).detach().cpu().numpy()
    if config['save_weights'] : model.save_model(save_model_path_for_current_dataset, filename = "model_MSE.pth")

    # Train and test SDTW
    config['recon_loss_type'] = 1
    loss_function = reconstruction_loss(config)
    model = MultiLayerPerceptron(layers, loss_function, config)
    model.fit(x_1_train, x_2_train, config)
    x_pred_SDTW = model(x_1_test).detach().cpu().numpy()
    if config['save_weights'] : model.save_model(save_model_path_for_current_dataset, filename = "model_SDTW.pth")

    # Train and test Pruned DTW
    config['recon_loss_type'] = 1
    config['bandwidth'] = bandwidth
    loss_function = reconstruction_loss(config)
    model = MultiLayerPerceptron(layers, loss_function, config)
    model.fit(x_1_train, x_2_train, config)
    x_pred_Pruned_DTW = model(x_1_test).detach().cpu().numpy()
    if config['save_weights'] : model.save_model(save_model_path_for_current_dataset, filename = "model_Pruned_SDTW.pth")

    # Train and test OTW
    config['recon_loss_type'] = 5
    loss_function = reconstruction_loss(config)
    model = MultiLayerPerceptron(layers, loss_function, config)
    model.fit(x_1_train, x_2_train, config)
    x_pred_OTW = model(x_1_test).detach().cpu().numpy()
    if config['save_weights'] : model.save_model(save_model_path_for_current_dataset, filename = "model_OTW.pth")

    # Train and test block SDTW
    config['recon_loss_type'] = 3
    loss_function = reconstruction_loss(config)
    model = MultiLayerPerceptron(layers, loss_function, config)
    model.fit(x_1_train, x_2_train, config)
    x_pred_block_SDTW = model(x_1_test).detach().cpu().numpy()
    if config['save_weights'] : model.save_model(save_model_path_for_current_dataset, filename = "model_block_SDTW.pth")

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Plot results

    ts_index = np.random.randint(0, x_1_test.shape[0])
    start_prediction = int(x_orig_test.shape[1] * config['portion_of_signals_for_input'])
    path_save = f"./Results/UCR_timeseries/{name_dataset}/time_series_{ts_index}.png"

    # x_pred_MSE = None
    # ts_index = 754
    # path_save = f"./Results/UCR_timeseries_PAPER/{name_dataset}_time_series_{ts_index}.pdf"

    x_pred_list = [x_pred_MSE, x_pred_SDTW, x_pred_Pruned_DTW, x_pred_block_SDTW, x_pred_OTW]
    x_pred_label = ['MSE', 'SDTW', 'Pruned_SDTW', 'Block_SDTW', 'OTW']

    dataset.visualize_prediction(ts_index, x_orig_test, start_prediction, x_pred_list, x_pred_label, path_save = path_save)

