"""
Simple multi layer perceptrons.
From https://tslearn.readthedocs.io/en/latest/auto_examples/autodiff/plot_soft_dtw_loss_for_pytorch_nn.html
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Imports

import torch

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

class MultiLayerPerceptron(torch.nn.Module):
    def __init__(self, layers, loss_function = None, config : dict = dict()):
        # At init, we define our layers
        super(MultiLayerPerceptron, self).__init__()
        self.layers = layers
        self.loss_function = loss_function
        
        lr = config['lr'] if 'lr' in config else 0.001

        self.optimizer = torch.optim.SGD(self.parameters(), lr = lr)
        # self.optimizer = torch.optim.Adamax(self.parameters(), lr = lr)
        # self.optimizer = torch.optim.AdamW(self.parameters(), lr = lr)

        if 'lr_decay_rate' in config :
            self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma = config['lr_decay_rate'])
        else :
            self.lr_scheduler = None

    def forward(self, X):
        # Convert to torch tensor
        if not isinstance(X, torch.Tensor):
            X = torch.Tensor(X)

        # Reshape for time series
        batch_size = X.size(0)
        X_reshaped = torch.reshape(X, (batch_size, -1))  

        # Forward step
        output = self.layers(X_reshaped)

        # Manipulations to deal with time series format
        output = torch.reshape(output, (batch_size, -1, 1))  

        return  output

    def fit(self, X, y, max_epochs = 10):
        # The fit method performs the actual optimization
        X_torch = torch.Tensor(X)
        y_torch = torch.Tensor(y)

        for e in range(max_epochs):
            self.optimizer.zero_grad()
            # Forward pass
            y_pred = self.forward(X_torch)

            # Compute Loss
            loss = self.loss_function.compute_loss(y_pred, y_torch)

            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            if self.lr_scheduler is not None : self.lr_scheduler.step()

            print("Epoch: {}\tLoss = {}".format(e + 1, loss))

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
