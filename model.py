"""
Simple multi layer perceptrons.
From https://tslearn.readthedocs.io/en/latest/auto_examples/autodiff/plot_soft_dtw_loss_for_pytorch_nn.html
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

import torch
import os

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

        self.training_failed = False

        # Save model configuration options
        self.save_model_path    = config['save_model_path'] if 'save_model_path' in config else "./saved_model/"
        self.model_name         = config['model_name'] if 'model_name' in config else "model.pth"
        self.save_every_n_epoch = config['save_every_n_epoch'] if 'save_every_n_epoch' in config else -1

    def forward(self, X, device = 'cpu'):
        # Convert to torch tensor and move to device if necessary
        if not isinstance(X, torch.Tensor):
            X = torch.Tensor(X).to(device)
            self.to(device) # Ensure the model is on the same device as the input

        # Reshape for time series
        batch_size = X.size(0)
        X_reshaped = torch.reshape(X, (batch_size, -1))
        
        # Forward step
        output = self.layers(X_reshaped)

        # Manipulations to deal with time series format
        # This is a workaround to ensure compatibility with the Soft-DTW loss function implementation
        output = torch.reshape(output, (batch_size, -1, 1))

        return output

    def fit(self, X, y, config : dict) :
        try :
            # The fit method performs the actual optimization
            X_torch = torch.Tensor(X).to(config['device'])
            y_torch = torch.Tensor(y).to(config['device'])
        
            if 'max_epochs' not in config : config['max_epochs'] = 100
            if 'batch_size' not in config : config['batch_size'] = 32
            if config['batch_size'] <= 0 or config['batch_size'] > X_torch.shape[0] : config['batch_size'] = X_torch.shape[0]
            if 'device' not in config : config['device'] = 'cpu'

            # Create DataLoader for batching
            dataset = torch.utils.data.TensorDataset(X_torch, y_torch)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size = config['batch_size'], shuffle=True)

            # Move model to the specified device
            self.to(config['device'])

            # Backup of model parameters in case of NaN during training
            backup_state_dict = self.state_dict()

            for e in range(config['max_epochs']):

                for i, (X_batch, y_batch) in enumerate(dataloader):
                    # Move data to the device
                    X_batch = X_batch
                    y_batch = y_batch

                    self.optimizer.zero_grad()
                    # Forward pass
                    y_pred = self.forward(X_batch)

                    # Compute Loss
                    loss = self.loss_function.compute_loss(y_pred, y_batch)

                    # Check if NaN is present in loss
                    if torch.isnan(loss).sum() > 0 :
                        # If NaN is detected, restore the paramaters from previous epoch and stop training

                        print("NaN detected in loss. Parameters restored from previous epoch. Training stopped.")
                        self.load_state_dict(backup_state_dict)

                        self.training_failed = True
                        return

                    # Backward pass
                    loss.backward()
                    self.optimizer.step()
                    
                    if self.lr_scheduler is not None : self.lr_scheduler.step()

                    backup_state_dict = self.state_dict()

                print("Epoch: {}\tLoss = {}".format(e + 1, loss))
                if self.save_every_n_epoch > 0 and (e + 1) % self.save_every_n_epoch == 0 : self.save_model(filename = f'{self.model_name}_epoch_{e + 1}.pth')

            self.training_failed = False
        except Error as e :
            print("ERROR")
            print(e)
            print("TRAIN STOPPED")

            self.training_failed = True

    def save_model(self, path : str = None, filename : str = None) :
        """
        Save the model to the path specified in self.save_model_path with the filename specified in self.model_name.
        If path is provided, it will override self.save_model_path.
        if filename is provided, it will override self.model_name.

        Parameters
        ----------
        path : str
            The directory path where the model will be saved. If None, uses self.save_model_path
            (By default, if not provided during initialization, it is self.save_model_path = "./saved_model/")
        filename : str
            The filename for the saved model. If None, uses self.model_name.pth
            (By default, if not provided during initialization, it is self.model_name = 'model')
        """
    
        if not os.path.exists(self.save_model_path) :
            os.makedirs(self.save_model_path)

        filename = filename if filename is not None else self.model_name
        filename += '.pth' if not filename.endswith('.pth') else ''

        full_path = os.path.join(self.save_model_path, filename)
        torch.save(self.state_dict(), full_path)
        print(f"Model saved to {full_path}")

    def load_model(self, path : str, filename : str = None) :
        """
        Load the model from the specified path.
        If filename is provided, it will be used as the filename. Otherwise, the 'model.pth' will be used.
        """
    
        if filename is None:
            filename = 'model.pth'

        full_path = os.path.join(path, filename)
        
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Model file {full_path} does not exist.")

        self.load_state_dict(torch.load(full_path))
        print(f"Model loaded from {full_path}")
        

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
