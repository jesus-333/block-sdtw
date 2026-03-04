"""
Advanced version of the of train_V1.py

@author: Alberto Zancanaro (Jesus)
@organization: Luxembourg Centre for Systems Biomedicine (LCSB)
"""
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

import numpy as np
from torch import nn

import dataset
from model import MultiLayerPerceptron
from block_sdtw import reconstruction_loss

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Settings

n_signals_to_generate = 120
length_1 = 150
length_2 = 100
bandwidth = 10

config = dict(
    # Training parameters
    batch_size = 30,
    lr = 0.001,                         # Learning rate (lr)
    epochs = 60,                        # Number of epochs to train the model
    use_scheduler = True,               # Use the lr scheduler
    lr_decay_rate = 0.999,              # Parameter of the lr exponential scheduler
    optimizer_weight_decay = 1e-2,      # Weight decay of the optimizer
    alpha = 1,                          # Multiplier of the reconstruction error
    recon_loss_type = 1,                # Loss function for the reconstruction (0 = L2, 1 = SDTW, 2 = SDTW-Divergence)
    block_size = 10,
    edge_samples_ignored = 0,           # Ignore this number of samples during the computation of the reconstructation loss
    gamma_dtw = 1,                      # Hyperparameter of the SDTW. Control the steepness of the soft-min inside the SDTW. The closer to 0 the closer the soft-min approximate the real min
    # device = "cuda" if torch.cuda.is_available() else "cpu",
    device = "cpu",
)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Dataset creation

# Get dataset
x_train, y_train, x_test, y_test = dataset.get_dataset()
x_subset = dataset.get_subset(x_train, y_train, n = 60)

# Generate input signals and the counterpart to predict (train)
x_1_train, x_2_train, x_orig_train, t_orig_train = dataset.generate_signals(x_subset, n_signals_to_generate = n_signals_to_generate , length_1 = length_1, length_2 = length_2)

# Generate input signals and the counterpart to predict (test)
x_1_test, x_2_test, x_orig_test, t_orig_test = dataset.generate_signals(x_test, n_signals_to_generate = n_signals_to_generate , length_1 = length_1, length_2 = length_2)

# Visualize randomly 4 pairs of signals
# for i in range(4) :
#     idx = np.random.randint(0, n_signals_to_generate)
#     dataset.visualize_signals(x_1_train[idx], x_2_train[idx], x_orig_train[idx], t_orig_train[idx])

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Layers definition

layers = nn.Sequential(
    nn.Linear(in_features = x_1_train.shape[1], out_features = 256),
    nn.GELU(),
    nn.Linear(in_features = 256, out_features = x_2_train.shape[1])
)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Model training

# Train and test MSE
config['recon_loss_type'] = 0
loss_function = reconstruction_loss(config)
model = MultiLayerPerceptron(layers, loss_function, config)
model.fit(x_1_train, x_2_train, config)
y_pred_MSE = model(x_1_test).detach().numpy()

# Train and test SDTW
config['recon_loss_type'] = 1
loss_function = reconstruction_loss(config)
model = MultiLayerPerceptron(layers, loss_function, config)
model.fit(x_1_train, x_2_train, config)
y_pred_SDTW = model(x_1_test).detach().numpy()

# Train and test block SDTW
config['recon_loss_type'] = 3
loss_function = reconstruction_loss(config)
model = MultiLayerPerceptron(layers, loss_function, config)
model.fit(x_1_train, x_2_train, config)
y_pred_block_SDTW = model(x_1_test).detach().numpy()

# Train and test PruneDTW
config['recon_loss_type'] = 1
config['bandwidth'] = bandwidth
loss_function = reconstruction_loss(config)
model = MultiLayerPerceptron(layers, loss_function, config)
model.fit(x_1_train, x_2_train, config)
y_pred_pruned_DTW = model(x_1_test).detach().numpy()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Plot results

ts_index = np.random.randint(0, n_signals_to_generate)
start_prediction = t_orig_test[ts_index] + length_1
dataset.visualize_prediction(ts_index, x_orig_test, start_prediction, y_pred_MSE, y_pred_SDTW, y_pred_block_SDTW)

