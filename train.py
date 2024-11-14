# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Imports

import numpy as np
from torch import nn
import matplotlib.pyplot as plt
from tslearn.datasets import CachedDatasets
import os

from model import MultiLayerPerceptron
from block_sdtw import reconstruction_loss

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Dataset and loss function declaration, training config

data_loader = CachedDatasets()
X_train, y_train, X_test, y_test = data_loader.load_dataset("Trace")

X_subset = X_train[y_train < 4]
np.random.shuffle(X_subset)
X_subset = X_subset[:50]

config = dict(
    # Training parameters
    batch_size = 30,
    lr = 0.001,                          # Learning rate (lr)
    epochs = 60,                       # Number of epochs to train the model
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

plot_and_save_all = False
loss_function = reconstruction_loss(config)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Layers defintion

layers = nn.Sequential(
    nn.Linear(in_features = 150, out_features = 256),
    nn.GELU(),
    nn.Linear(in_features = 256, out_features = 125)
)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Model training

# Train and test MSE
config['recon_loss_type'] = 0
loss_function = reconstruction_loss(config)
model = MultiLayerPerceptron(layers, loss_function, config)
model.fit(X_subset[:, :150], X_subset[:, 150:], max_epochs = config['epochs'])
y_pred_MSE = model(X_test[:, :150, 0]).detach().numpy()

# Train and test SDTW
config['recon_loss_type'] = 1
loss_function = reconstruction_loss(config)
model = MultiLayerPerceptron(layers, loss_function, config)
model.fit(X_subset[:, :150], X_subset[:, 150:], max_epochs = config['epochs'])
y_pred_SDTW = model(X_test[:, :150, 0]).detach().numpy()

# Train and test block SDTW
config['recon_loss_type'] = 3
loss_function = reconstruction_loss(config)
model = MultiLayerPerceptron(layers, loss_function, config)
model.fit(X_subset[:, :150], X_subset[:, 150:], max_epochs = config['epochs'])
y_pred_block_SDTW = model(X_test[:, :150, 0]).detach().numpy()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Plot results

# Plot function
def plot_prediction(ts_index, X_test, y_pred_MSE, y_pred_SDTW, y_pred_block_SDTW, show_figure = True): 
    # Change backend
    plt.switch_backend('TkAgg')

    fig, ax = plt.subplots(1, 1, figsize = (10, 10))

    ax.plot(X_test[ts_index].ravel(), label = 'True')
    ax.plot(np.arange(150, 275), y_pred_MSE[ts_index], 'r-', label = 'MSE', color = 'g')
    ax.plot(np.arange(150, 275), y_pred_SDTW[ts_index], 'r-', label = 'SDTW', color = 'b')
    ax.plot(np.arange(150, 275), y_pred_block_SDTW[ts_index], 'r-', label = 'Block SDTW', color = 'r')
    ax.set_title('Prediction of the time series ' + str(ts_index))

    ax.grid()
    ax.legend()

    fig.tight_layout()
    if show_figure : fig.show()

    return fig, ax


if plot_and_save_all:
    path_save = 'Results/'
    os.makedirs(path_save, exist_ok = True)

    # Iterate over the test set
    for i in range(len(X_test)):
        print('Plotting prediction of the time series ' + str(i))
        fig, ax = plot_prediction(i, X_test, y_pred_MSE, y_pred_SDTW, y_pred_block_SDTW, show_figure = False)

        fig.savefig(path_save + 'prediction_' + str(i) + '.png')
        plt.close()
else :
    plot_prediction(0, X_test, y_pred_MSE, y_pred_SDTW, y_pred_block_SDTW, show_figure = True)
    plot_prediction(50, X_test, y_pred_MSE, y_pred_SDTW, y_pred_block_SDTW, show_figure = True)
    plot_prediction(87, X_test, y_pred_MSE, y_pred_SDTW, y_pred_block_SDTW, show_figure = True)


