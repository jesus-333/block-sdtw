# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Imports

import numpy as np
from torch import nn
import matplotlib.pyplot as plt
from tslearn.datasets import CachedDatasets

from model import MultiLayerPerceptron
from block_sdtw import reconstruction_loss

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Dataset and loss function declaration

data_loader = CachedDatasets()
X_train, y_train, X_test, y_test = data_loader.load_dataset("Trace")

X_subset = X_train[y_train < 4]
np.random.shuffle(X_subset)
X_subset = X_subset[:50]

config = dict(
    batch_size = 30,
    lr = 1e-2,                          # Learning rate (lr)
    epochs = 1000,                      # Number of epochs to train the model
    use_scheduler = True,               # Use the lr scheduler
    lr_decay_rate = 0.999,              # Parameter of the lr exponential scheduler
    optimizer_weight_decay = 1e-2,      # Weight decay of the optimizer
    alpha = 1,                          # Multiplier of the reconstruction error
    recon_loss_type = 1,                # Loss function for the reconstruction (0 = L2, 1 = SDTW, 2 = SDTW-Divergence)
    block_size = 15,
    edge_samples_ignored = 0,           # Ignore this number of samples during the computation of the reconstructation loss
    gamma_dtw = 1,                      # Hyperparameter of the SDTW. Control the steepness of the soft-min inside the SDTW. The closer to 0 the closer the soft-min approximate the real min
    # device = "cuda" if torch.cuda.is_available() else "cpu",
    device = "cpu",
)

loss_function = reconstruction_loss(config)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Layers defintion

layers = nn.Sequential(
    nn.Linear(in_features = 150, out_features = 256),
    nn.ReLU(),
    nn.Linear(in_features = 256, out_features = 125)
)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Model training

# Train and test MSE
config['recon_loss_type'] = 0
loss_function = reconstruction_loss(config)
model = MultiLayerPerceptron(layers, loss_function)
model.fit(X_subset[:, :150], X_subset[:, 150:], max_epochs = config['epochs'])
ts_index = 50
y_pred_MSE = model(X_test[:, :150, 0]).detach().numpy()

# Train and test SDTW
config['recon_loss_type'] = 1
loss_function = reconstruction_loss(config)
model = MultiLayerPerceptron(layers, loss_function)
model.fit(X_subset[:, :150], X_subset[:, 150:], max_epochs = config['epochs'])
ts_index = 50
y_pred_SDTW = model(X_test[:, :150, 0]).detach().numpy()

# Train and test block SDTW
config['recon_loss_type'] = 3
loss_function = reconstruction_loss(config)
model = MultiLayerPerceptron(layers, loss_function)
model.fit(X_subset[:, :150], X_subset[:, 150:], max_epochs = config['epochs'])
ts_index = 50
y_pred_block_SDTW = model(X_test[:, :150, 0]).detach().numpy()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Plot

# Change backend
plt.switch_backend('TkAgg')

fig, ax = plt.subplots(1, 1, figsize = (10, 10))

ax.plot(X_test[ts_index].ravel(), label = 'True')
ax.plot(np.arange(150, 275), y_pred_MSE[ts_index], 'r-', label = 'MSE', color = 'g')
ax.plot(np.arange(150, 275), y_pred_SDTW[ts_index], 'r-', label = 'SDTW', color = 'b')
ax.plot(np.arange(150, 275), y_pred_block_SDTW[ts_index], 'r-', label = 'Block SDTW', color = 'r')

ax.grid()
ax.legend()

fig.tight_layout()
fig.show()
