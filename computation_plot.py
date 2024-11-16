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

t_list = (np.arange(20) + 2) * 50

loss_type_list = [1, 3]

plot_config = dict(
    figsize = (15, 8),
    fontsize = 20,
    normalization = -1,
    save_fig = False
)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 


# loss_type_str = 'SDTW_rust' if loss_type_to_use == 0 else 'SDTW_standard' if loss_type_to_use == 1 else 'SDTW_divergence' if loss_type_to_use == 2 else 'SDTW_block' if loss_type_to_use == 3 else 'SDTW_block_divergence'

