"""
Plot the results of the benchmark comparing the various sdtw implementations.
The results are obtained with the script ``./benchmark/compute_sdtw_implementations.py``

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

implementation_list = ['mag', 'pysdtw', 'ron']

# x-axis settings.
T_list = (np.arange(10) + 1) * 100

# Benchmark settings to plot (must be the same, or a subset, of the settings used in the script that computes the benchmark)
B_list = [10, 100]
C_list = [1]

# Original benchmark settings
# B_list = [1, 10, 100]
# C_list = [1, 2, 24]

# Path where the results of the benchmark are saved (must be the same as in the script that computes the benchmark)
path_results = 'benchmark/MAC_M4_PRO/'

plot_config = dict(
    figsize = (20, 12),
    fontsize = 18,
    marker = 'o',
    markersize = 10,
    linewidth = 2,
    y_scale_log = True,
    use_milliseconds = True,
    path_save = 'benchmark/MAC_M4_PRO/sdtw_implementations.png',
)

# Color settings for the plot. The keys must be the same as the labels of the plot.
# Note that in this case I run the script a first time to get the labels with the current settings and then I copy-paste them here and set the colors.
color_dict = {
    'sdtw_mag (B=10, C=1)'            : 'green',
    'sdtw_mag (B=100, C=1)'           : 'darkgreen',
    'sdtw_pysdtw (B=10, C=1)'         : 'violet',
    'sdtw_pysdtw (B=100, C=1)'        : 'darkviolet',
    'sdtw_ron (B=10, C=1)'            : 'pink',
    'sdtw_ron (B=100, C=1)'           : 'deeppink',
}

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Load data

# Load raw benchmark results from file
results_dict = benchmark.load_benchmark_results(path_results)

# Variable to store data to plot and labels
data_to_plot_mean = []
data_to_plot_std = []
labels_to_plot = []

# Create SDTW loss functions
for implementation in implementation_list :
    loss_function_name = f'sdtw_{implementation}'
    
    data_sdtw_mean, data_sdtw_std, labels_sdtw = benchmark.get_array_to_plot(results_dict, loss_function_name, 'T', B_list, T_list, C_list)
    data_to_plot_mean += data_sdtw_mean
    data_to_plot_std  += data_sdtw_std
    labels_to_plot    += labels_sdtw

# Sort data in alphabetical order of labels (this is useful to have the same order of colors in the plot)
idx_sort = np.argsort(labels_to_plot)
data_to_plot_mean = [data_to_plot_mean[i] for i in idx_sort]
data_to_plot_std = [data_to_plot_std[i] for i in idx_sort]
labels_to_plot = [labels_to_plot[i] for i in idx_sort]
    
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Plot results

fig, ax = benchmark.plot_benchmark(plot_config, T_list, 'Time Samples T', data_to_plot_mean, labels_to_plot, color_dict = color_dict)
plt.show()

