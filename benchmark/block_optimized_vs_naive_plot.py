"""
Plot the results of the benchmark comparing the optimized and naive implementations of block dtw.
The results are obtained with the script ``benchmark/block_optimized_vs_naive_compute.py``.

Note that this is only an example script.

Authors
-------
Alberto Zancanaro <alberto.zancanaro@uni.lu>
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

import numpy as np
import matplotlib.pyplot as plt
import torch

from dtw_loss_functions import benchmark

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Settings

# Torch settings
use_cuda = torch.cuda.is_available()
device = 'cuda' if use_cuda and use_cuda else 'cpu'

# x-axis settings.
T_list = (np.arange(10) + 1) * 100

# Benchmark settings to plot (must be the same, or a subset, of the settings used in the script that computes the benchmark)
B_list = [10, 100]
C_list = [1]
block_size_list = [10, 100]

# Original benchmark settings
# B_list = [1, 10, 100]
# C_list = [1, 2, 24]
# block_size_list = [10, 50, 100]

# Path where the results of the benchmark are saved (must be the same as in the script that computes the benchmark)
path_results = 'benchmark/MAC_M4_PRO/'

plot_config = dict(
    figsize = (20, 12),
    fontsize = 18,
    y_scale_log = False,
    add_std = False,
)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Load data

# Load raw benchmark results from file
results_dict = benchmark.load_benchmark_results(path_results)

# Variable to store data to plot and labels
data_to_plot_mean = []
data_to_plot_std = []
labels_to_plot = []

# Create array suitable for plotting
for block_size in block_size_list :
    # Get naive results
    loss_function_name = f'block_naive_{block_size}'
    data_block_naive_mean, data_block_naive_std, labels_naive = benchmark.get_array_to_plot(results_dict, loss_function_name, 'T', B_list, T_list, C_list)

    # Save naive results for plotting
    data_to_plot_mean += data_block_naive_mean
    data_to_plot_std  += data_block_naive_std
    labels_to_plot    += labels_naive

    # Get optimized results
    loss_function_name = f'block_optimized_{block_size}'
    data_block_optimized_mean, data_block_optimized_std, labels_optimized = benchmark.get_array_to_plot(results_dict, loss_function_name, 'T', B_list, T_list, C_list)

    # Save optimized results for plotting
    data_to_plot_mean += data_block_optimized_mean
    data_to_plot_std  += data_block_optimized_std
    labels_to_plot    += labels_optimized
    
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Plot results

fig, ax = benchmark.plot_benchmark(plot_config, T_list, 'Time Samples T', data_to_plot_mean, labels_to_plot)
plt.show()
