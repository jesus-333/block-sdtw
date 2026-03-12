"""
Use the ``benchmark`` module inside the package to evaluate the performance of the block-optimized implementation  against the naive one.

Authors
-------
Alberto Zancanaro <alberto.zancanaro@uni.lu>
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

import numpy as np
import torch

from dtw_loss_functions import benchmark, block_dtw

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

block_size_list = [10, 50, 100]

# SDTW config to use inside block dtw
implementation = 'mag'
sdtw_config = dict(
    use_cuda = use_cuda,
    normalize = False,
)

# Path to save the results
path_save = 'benchmark/MAC_M4_PRO/'

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Benchmark

for i in range(len(block_size_list)) :
    block_size = block_size_list[i]

    # Create loss function instances
    loss_block_naive = block_dtw.block_dtw_naive(block_size = block_size, implementation = implementation, sdtw_config = sdtw_config)
    loss_block_optimized = block_dtw.block_dtw_optimized(block_size = block_size, implementation = implementation, sdtw_config = sdtw_config)
    
    # Add the loss functions to the dictionary of loss functions to test
    loss_functions_to_use = dict()
    loss_functions_to_use[f'block_naive_{block_size}'] = loss_block_naive
    loss_functions_to_use[f'block_optimized_{block_size}'] = loss_block_optimized

    # Compute benchmark
    benchmark.compute_benchmark(B_list, T_list, C_list, loss_functions_to_use, device, n_repetitions, path_save, print_progress)
