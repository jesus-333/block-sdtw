"""
Since there are various implementations online of the SoftDTW, this module was created as a wrapper to be able to easily switch between different implementations.

Currently, the following implementations are available :

- pytorch-softdtw-cuda by Mehran Maghoumi :cite:`soft_dtw_mag_1` :cite:`soft_dtw_mag_2`
    - GitHub repository : https://github.com/Maghoumi/pytorch-softdtw-cuda
- pysdtw by Antoine Loriette
    - GitHub repository : https://github.com/toinsson/pysdtw
    - PyPi Page : https://pypi.org/project/pysdtw/
- sdtw-cuda-torch by BGU-CS-VIL (implemented by Ron Shapira Weber) :cite:`soft_dtw_ron_1` :cite:`soft_dtw_ron_2`
    - GitHub repository : https://github.com/BGU-CS-VIL/sdtw-cuda-torch

If you use this module, please cite together with this package the original paper of the implementation you are using.

Example
-------

Mehran Maghoumi's (``mag``) implementation

>>> import torch
>>> from dtw_loss_functions import soft_dtw
>>> use_cuda = torch.cuda.is_available()
>>> sdtw_loss = soft_dtw.soft_dtw(implementation = 'mag', sdtw_config = {'use_cuda' : use_cuda, 'gamma' : 0.1})
>>> batch_size = 5
>>> time_samples = 300
>>> channels = 1
>>> device = 'cuda' if use_cuda else 'cpu'
>>> x   = torch.randn(batch_size, time_samples, channels).to(device)
>>> x_r = torch.randn(batch_size, time_samples, channels).to(device)
>>> output_sdtw = sdtw_loss(x, x_r)

Ron Shapira Weber's (``ron``) implementation

>>> import torch
>>> from dtw_loss_functions import soft_dtw
>>> sdtw_loss = soft_dtw.soft_dtw(implementation = 'ron', sdtw_config = {'gamma' : 0.1, 'dist' : 'sqeuclidean'})
>>> batch_size = 5
>>> time_samples = 300
>>> channels = 1
>>> device = 'cuda' if torch.cuda.is_available() else 'cpu'
>>> x   = torch.randn(batch_size, time_samples, channels).to(device)
>>> x_r = torch.randn(batch_size, time_samples, channels).to(device)
>>> output_sdtw = sdtw_loss(x, x_r)

Authors
-------
Alberto Zancanaro <alberto.zancanaro@uni.lu>

"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

import torch

import pysdtw
from .soft_dtw_implementations import soft_dtw_cuda_mag, pysdtw_normalize, soft_dtw_cuda_ron

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

class soft_dtw(torch.nn.Module) :
    """
    SoftDTW class. This class is a wrapper for the different implementations of the SoftDTW.
    The implementation can be selected by passing the ``implementation`` argument to the constructor. The available implementations are:

    - `mag`: pytorch-softdtw-cuda by Mehran Maghoumi
    - `pysdtw`: pysdtw by Antoine Loriette
    - `ron` : sdtw-cuda-torch by BGU-CS-VIL (implemented by Ron Shapira Weber)

    Together with the implementation, a configuration dictionary can be passed to the constructor to set the parameters of the SDTW function. 
    The available parameters depend on the implementation selected, but some parameters are common for all implementations (e.g. ``gamma``, ``normalize``, ``bandwidth``). 
    The available parameters are listed below.

    Parameters
    ----------
    implementation : str
        Implementation to use for the SDTW.

        - ``mag``. Use the implementation by Mehran Maghoumi :cite:`soft_dtw_mag_1` :cite:`soft_dtw_mag_2`.
        - ``pysdtw``. Use the implementation in the pysdtw package by Antoine Loriette.
        - ``ron`` Use the implementation by Ron Shapira Weber :cite:`soft_dtw_ron_1` :cite:`soft_dtw_ron_2`.

    sdtw_config : dict, optional

        Configuration dictionary for the SDTW function.
        Note that if a parameter is not specified in the configuration dictionary, the default value will be used (the default values are specified in the description of each parameter).
        The dictionary can contain the following keys :

            use_cuda : bool
                If ``True``, this class will use the CUDA implementation of the SDTW. Only for the ``mag`` and ``pysdtw`` implementations. Default is ``False``.
            gamma : float, optional
                Value of the gamma hyperparameter for the SDTW. Default is ``1``.
            normalize : bool, optional
                If ``True``, the SDTW divergence will be computed instead of the SDTW. Default is ``False``.
            bandwidth : float, optional
                Sakoe-Chiba bandwidth for pruning. If the ``None`` is given, no pruning is applied. Default is ``None``.
            dist_func : function, optional
                Only for the ``mag`` and ``pysdtw`` implementations. If passed, this function will be used as distance function to use for the SDTW.
                If ``None``, the default distance function of the implementation will be used (squared Euclidean distance for both implementations).
                Default is ``None``.
            dist : str, optional
                Only for the ``ron`` implementation. It has the same purpose as ``dist_func`` for the other implementations, but in this case must be a string.
            fused : bool, optional
                Only for the ``ron`` implementation.

                - ``None``  -> auto (use fused only when possible)
                - ``True``  -> require fused (error if not possible)
                - ``False`` -> never fused (always materialize D and use D-based autograd)
    """

    def __init__(self, implementation : str = 'mag', sdtw_config : dict = {}) :
        super().__init__()

        # Check if the selected implementation is valid
        self.set_implementation(implementation, reset_sdtw_function = False)

        # Set the configuration for the SDTW function
        self.set_sdtw_config(sdtw_config, reset_sdtw_function = False)
        
        # Get the SDTW function based on the implementation selected and the configuration specified
        self.sdtw_function = self.create_sdtw_function(sdtw_config)

    def forward(self, x : torch.Tensor, y : torch.Tensor) -> torch.Tensor :
        """
        Compute the SoftDTW distance between two time series.

        Parameters
        ----------
        x : torch.Tensor
            First input tensor of shape B x T x C
        y : torch.Tensor
            Second input tensor of shape B x T x C

        Returns
        -------
        torch.Tensor
            SoftDTW distance between the two time series
        """

        return self.sdtw_function(x, y)

    def set_implementation(self, implementation : str, reset_sdtw_function : bool = True) :
        """
        Set the implementation to use for the SDTW. This function can be used to change the implementation after the class has been initialized.
        """
        self.check_implementation(implementation)
        self.implementation = implementation

    def set_sdtw_config(self, sdtw_config : dict = {}, reset_sdtw_function : bool = True) :
        """
        Set the configuration for the SDTW function. This function can be used to change the configuration after the class has been initialized.

        Parameters
        ----------
        sdtw_config : dict
            Configuration dictionary for the SDTW function. The keys of the dictionary are the same as the parameters of the constructor.
            Note that if a key is absent from the dictionary, the default value for that parameter will be used (the default values are specified in the description of each parameter).
        reset_sdtw_function : bool, optional
            If ``True``, the SDTW function will be reset with the new configuration.
            If ``False``, the SDTW function will not be reset, but the new configuration will be saved as an attribute of the class.
            Default is ``True``.
        """

        # Parameters for the SDTW (Common for all implementations)
        # Note that the get method is used to set the default values for the parameters, in case they are not specified in the configuration dictionary
        self.gamma = sdtw_config.get('gamma', 1)
        self.normalize = sdtw_config.get('normalize', False)
        self.bandwidth = sdtw_config.get('bandwidth', None)

        # Saved the parameters specific for each implementation as attributes of the class, in case they are needed later (e.g. for the backward pass)
        if self.implementation == 'mag' or self.implementation == 'pysdtw' :
            self.use_cuda = sdtw_config.get('use_cuda', False)
            self.dist_func = sdtw_config.get('dist_func', None)
        elif self.implementation == 'ron' :
            self.dist = sdtw_config.get('dist', 'sqeuclidean')
            self.fused = sdtw_config.get('fused', None)

        # Reset the SDTW function with the new configuration
        if reset_sdtw_function :
            self.sdtw_function = self.get_sdtw_function(sdtw_config)
    
    def create_sdtw_function(self, sdtw_config : dict) :
        """
        Create and return an istance of the SDTW function based on the current implementation and parameters.

        Parameters
        ----------
        sdtw_config : dict
            Configuration dictionary for the SDTW function. The keys of the dictionary are the same as the parameters of the constructor.
        """

        if self.implementation is None :
            raise ValueError("Implementation not set. Please set the implementation before instantiating the SDTW function.")
        else :
            if self.implementation == 'mag' : 
                sdtw_function = soft_dtw_cuda_mag.SoftDTW(use_cuda = self.use_cuda, gamma = self.gamma, normalize = self.normalize, bandwidth = self.bandwidth, dist_func = self.dist_func)
            elif self.implementation == 'pysdtw' and self.normalize : 
                sdtw_function = pysdtw_normalize.pysdtw_normalized(use_cuda = self.use_cuda, gamma = self.gamma, bandwidth = self.bandwidth, dist_func = self.dist_func)
            elif self.implementation == 'pysdtw' and not self.normalize :
                sdtw_function = pysdtw.SoftDTW(use_cuda = self.use_cuda, gamma = self.gamma, bandwidth = self.bandwidth, dist_func = self.dist_func)
            elif self.implementation == 'ron' :
                sdtw_function = soft_dtw_cuda_ron.SoftDTW(gamma = self.gamma, normalize = self.normalize, bandwidth = self.bandwidth, dist = self.dist, fused = self.fused)

            return sdtw_function

    def check_implementation(self, implementation : str) :
        """
        Check if the selected implementation is valid. If not, raise an error.
        """

        implementations = ['mag', 'pysdtw', 'ron']

        if implementation not in implementations :
            raise ValueError(f"Invalid implementation selected. Implementations available: {implementations}. Selected implementation: {implementation}.")
