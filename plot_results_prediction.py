"""
Load the weights and print the prediction results
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from tslearn.datasets import CachedDatasets
import os

from model import MultiLayerPerceptron

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

path_weights = "saved_model/experiment_3/"
plot_and_save_all = True

plot_config = dict(
    figsize = (12, 9),
    fontsize = 25,
    add_title = False,
    add_legend = True,
    linewidth = 2,
    format_list = ['png', 'pdf', 'eps']
)


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

data_loader = CachedDatasets()
X_train, y_train, X_test, y_test = data_loader.load_dataset("Trace")

X_subset = X_train[y_train < 4]
np.random.shuffle(X_subset)
X_subset = X_subset[:50]


X_subset = X_train
np.random.shuffle(X_subset)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Functions

def get_layer() :
    layers = nn.Sequential(
        nn.Linear(in_features = 150, out_features = 256),
        nn.SELU(),
        nn.Linear(in_features = 256, out_features = 1024),
        nn.SELU(),
        nn.Linear(in_features = 1024, out_features = 1024),
        nn.SELU(),
        nn.Linear(in_features = 1024, out_features = 1024),
        nn.SELU(),
        nn.Linear(in_features = 1024, out_features = 125)
    )
    
    nerons_hidden_layer = 512
    layers = nn.Sequential(
        nn.Linear(in_features = 150, out_features = nerons_hidden_layer),
        # nn.BatchNorm1d(nerons_hidden_layer),
        nn.SELU(),
        nn.Linear(in_features = nerons_hidden_layer, out_features = 125)
    )

    return layers

# Plot function
def plot_prediction(ts_index, X_test,
                    y_pred_MSE, y_pred_SDTW, y_pred_Pruned_DTW, y_pred_block_SDTW,
                    plot_config, show_figure = True):
    # Change backend
    plt.switch_backend('TkAgg')

    fig, ax = plt.subplots(1, 1, figsize = plot_config['figsize'])

    ax.plot(X_test[ts_index].ravel(), label = 'True')
    # ax.plot(np.arange(150, 275), y_pred_MSE[ts_index], label = 'MSE', color = 'magenta', linewidth = plot_config['linewidth'])
    ax.plot(np.arange(150, 275), y_pred_SDTW[ts_index], label = 'SDTW', color = 'tab:orange', linewidth = plot_config['linewidth'])
    ax.plot(np.arange(150, 275), y_pred_Pruned_DTW[ts_index], label = 'Pruned DTW (bandwidth = 10)', color = 'tab:purple', linewidth = plot_config['linewidth'])
    ax.plot(np.arange(150, 275), y_pred_block_SDTW[ts_index], label = 'Block SDTW', color = 'tab:red', linewidth = plot_config['linewidth'])
    if plot_config['add_title'] : ax.set_title('Prediction of the time series ' + str(ts_index), fontsize = plot_config['fontsize'])
    ax.axvline(x = 150, color = 'black', linestyle = '--')

    ax.grid()
    if plot_config['add_legend'] : ax.legend(fontsize = plot_config['fontsize'])
    ax.set_ylim(X_test[ts_index].min() * 1.1, X_test[ts_index].max() * 1.1)
    ax.set_xlim([0, 275])
    ax.tick_params(axis = 'both', which = 'major', labelsize = plot_config['fontsize'])

    fig.tight_layout()
    if show_figure : fig.show()

    return fig, ax

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

model = MultiLayerPerceptron(get_layer())

# Prediction MSE
model.load_state_dict(torch.load(path_weights + "model_MSE.pth"))
y_pred_MSE = model(X_test[:, :150]).detach().numpy()

# Prediction SDTW
model.load_state_dict(torch.load(path_weights + "model_SDTW.pth"))
y_pred_SDTW = model(X_test[:, :150]).detach().numpy()

# Prediction Pruned DTW
model.load_state_dict(torch.load(path_weights + "model_prunedDTW.pth"))
y_pred_Pruned_DTW = model(X_test[:, :150]).detach().numpy()

# Prediction block_SDTW
model.load_state_dict(torch.load(path_weights + "model_block_SDTW.pth"))
y_pred_block_SDTW = model(X_test[:, :150]).detach().numpy()

if plot_and_save_all:
    path_save = 'Results/trace_timeseries_prediction/'
    os.makedirs(path_save, exist_ok = True)

    # Iterate over the test set
    for i in range(len(X_test)):
        print('Plotting prediction of the time series ' + str(i))
        fig, ax = plot_prediction(i, X_test, y_pred_MSE, y_pred_SDTW, y_pred_Pruned_DTW, y_pred_block_SDTW, plot_config, show_figure = False)

        for j in range(len(plot_config['format_list'])):

            fig.savefig(path_save + 'prediction_' + str(i) + '.' + plot_config['format_list'][j])

        plt.close()
else :
    plot_prediction(0, X_test, y_pred_MSE, y_pred_SDTW, y_pred_block_SDTW, plot_config, show_figure = True)
    plot_prediction(50, X_test, y_pred_MSE, y_pred_SDTW, y_pred_block_SDTW, plot_config, show_figure = True)
    plot_prediction(87, X_test, y_pred_MSE, y_pred_SDTW, y_pred_block_SDTW, plot_config, show_figure = True)








