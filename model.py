"""
Simple multi layer perceptrons.
From https://tslearn.readthedocs.io/en/latest/auto_examples/autodiff/plot_soft_dtw_loss_for_pytorch_nn.html
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Imports

import torch

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

class MultiLayerPerceptron(torch.nn.Module):
    def __init__(self, layers, loss_function = None):
        # At init, we define our layers
        super(MultiLayerPerceptron, self).__init__()
        self.layers = layers
        self.loss_function = loss_function
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.001)

    def forward(self, X):
        # The forward method informs about the forward pass: how one computes outputs of the network
        # from the input and the parameters of the layers registered at init
        if not isinstance(X, torch.Tensor):
            X = torch.Tensor(X)
        batch_size = X.size(0)
        X_reshaped = torch.reshape(X, (batch_size, -1))  # Manipulations to deal with time series format
        output = self.layers(X_reshaped)
        return torch.reshape(output, (batch_size, -1, 1))  # Manipulations to deal with time series format

    def fit(self, X, y, max_epochs=10):
        # The fit method performs the actual optimization
        X_torch = torch.Tensor(X)
        y_torch = torch.Tensor(y)

        for e in range(max_epochs):
            self.optimizer.zero_grad()
            # Forward pass
            y_pred = self.forward(X_torch)
            print(y_pred.shape)
            print(y_torch.shape)

            # Compute Loss
            loss = self.loss_function.compute_loss(y_pred, y_torch)

            # Backward pass
            loss.backward()
            self.optimizer.step()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
