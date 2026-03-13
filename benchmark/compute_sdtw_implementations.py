"""
Use the ``benchmark`` module inside the package to evaluate the performance of the various sdtw implementations.

Authors
-------
Alberto Zancanaro <alberto.zancanaro@uni.lu>
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

import numpy as np
import torch

from dtw_loss_functions import benchmark, block_dtw, soft_dtw

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Settings

# Torch settings
use_cuda = torch.cuda.is_available()
device = 'cuda' if use_cuda and use_cuda else 'cpu'

# Benchmark settings
n_repetitions = 100
T_list = (np.arange(10) + 1) * 100
B_list = [1, 10, 100]
C_list = [1, 2, 24]
print_progress = True

# SDTW config to use inside block dtw
implementation_list = ['mag', 'pysdtw', 'ron']
sdtw_config = dict(
    use_cuda = use_cuda,
    normalize = False,
)

# Path to save the results
path_save = 'benchmark/MAC_M4_PRO/'

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Benchmark

loss_functions_to_use = dict()

# Create SDTW loss functions
for implementation in implementation_list :
    loss_sdtw = soft_dtw.soft_dtw(implementation = implementation, sdtw_config = sdtw_config)
    loss_functions_to_use[f'sdtw_{implementation}'] = loss_sdtw

# Compute benchmark
benchmark.compute_benchmark(B_list, T_list, C_list, loss_functions_to_use, device, n_repetitions, path_save, print_progress)
