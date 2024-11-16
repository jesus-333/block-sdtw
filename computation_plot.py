"""
Plot the results obtained with computation_evaluation.py
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Imports

import matplotlib.pyplot as plt
import numpy as np

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Settings

path_files = "Results/computation_time/raspberry/"

t_list = (np.arange(19) + 2) * 50

loss_type_list = [1, 3]

plot_config = dict(
    figsize = (15, 8),
    fontsize = 20,
    normalization = -1,
    save_fig = False
)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

label_list = []
results_matrix = []

for i in range(len(loss_type_list)) : # Iterate over loss function variation
    loss_type = loss_type_list[i]
    loss_type_str = 'SDTW_rust' if loss_type == 0 else 'SDTW_standard' if loss_type == 1 else 'SDTW_divergence' if loss_type == 2 else 'SDTW_block' if loss_type == 3 else 'SDTW_block_divergence'

    label_list.append(loss_type_str)

    for j in range(len(t_list)) : # Iterate over sequence length
        t = t_list[j]



