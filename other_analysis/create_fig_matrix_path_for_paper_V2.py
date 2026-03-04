"""
Create the figures that show ipotetical paht in the DTW matrix for the following algorithm
- DTW
- PrunedDTW
- BlockDTW

@author: Alberto Zancanaro (Jesus)
@organization: Luxembourg Centre for Systems Biomedicine (LCSB)
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

import matplotlib.pyplot as plt
import numpy as np
import os
import dtw

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Settings

n_samples = 300
block_size = 60

f1 = 15
f2 = 30

# If True add noise to the time series. The noise is added as a random value drawn from a normal distribution with mean 0 and standard deviation equal to noise_strength.
add_noise = True
noise_strength = 0.5

plot_config = dict(
    figsize = (10, 10),
    aspect = 'equal',
    color_s1 = 'blue',
    color_s2 = 'orange',
    path_save = './other_analysis/figures/path_2/'
)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# Check that the number of samples is divisible by the block size
# The plot (IMHO) looks better if all the blocks have the same size
if n_samples % block_size != 0 : raise ValueError(f"The number of samples ({n_samples}) must be divisible by the block size ({block_size}).")

def plot_dtw_matrix(dtw_matrix : np.ndarray, s1 : np.ndarray, s2 : np.ndarray, t, plot_config : dict, matrix_overlap = False) :
    """
    This function plot the DTW matrix with the path and the two time series on the left and on the top of the matrix.
    If the parameter matrix_overlap is False, the cells of the matrix will be considered as follows :
    - 0 : cell where it is possible to go (light red)
    - 1 : cell where the path goes through (black)
    - 2 : cell where it is not possible to go (white)
    

    If the parameter matrix_overlap is True, the cells of the matrix will be colored with a colormap that shows the overlap between the path and the possible cells.
    - 0 : cell not part of the path (white)
    - 1 : cell part of first path (black)
    - 2 : cell part of second path (red)
    - 3 : cell part of both paths (green)

    Parameters
    ----------
    dtw_matrix : np.ndarray
        The DTW matrix to plot.
    s1 : np.ndarray
        The first time series.
    s2 : np.ndarray
        The second time series.
    t : np.ndarray
        The time points corresponding to the time series.
    plot_config : dict
        The configuration for the plot, with the following keys :
        - figsize : tuple, the size of the figure
        - aspect : str, the aspect ratio of the plot
        - color_s1 : str, the color of the first time series
        - color_s2 : str, the color of the second time series
    matrix_overlap : bool, optional
        If True, the cells of the matrix will be colored with a colormap that shows the overlap between the path and the possible cells.
    """


    # Create a colormap
    if matrix_overlap :
        cmap_colors = np.zeros((4, 4)) # 3 colors, 4 channels (RGBA)
        cmap_colors[0] = [1, 1, 1, 1]      # White for cell where there is no path (0)
        cmap_colors[1] = [0, 0, 0, 1]   # Black for cell where the path goes through (1)
        cmap_colors[2] = [1, 0, 0, 1]      # Red for cell where there is the second path (2)
        cmap_colors[3] = [0, 1, 0, 1]      # Green for cell where there is both paths (3)
    else :
        cmap_colors = np.zeros((3, 4)) # 3 colors, 4 channels (RGBA)
        cmap_colors[0] = [1, 0.8, 0.8, 1]  # Light red for cell where it is possible to go (0)
        cmap_colors[1] = [0, 0, 0, 1]   # Black for cell where the path goes through (1)
        cmap_colors[2] = [1, 1, 1, 1]      # White for cell where it is not possible to go (2)
    custom_cmap = plt.matplotlib.colors.ListedColormap(cmap_colors)

    fig, ax = plt.subplots(2, 2, figsize = plot_config['figsize'], gridspec_kw = {'width_ratios': [1, 4], 'height_ratios': [4, 1]})

    ax[0, 1].imshow(dtw_matrix.T, cmap = custom_cmap, aspect = plot_config['aspect'])
    ax[0, 1].set_xticks([])
    ax[0, 1].set_yticks([])

    ax[0, 0].plot(s1, t, color = plot_config['color_s1'])
    ax[0, 0].set_xticks([])
    ax[0, 0].set_yticks([])
    ax[0, 0].set_ylim(t[-1], t[0])
    # ax[0, 0].set_xlim([1 + noise_strength, - 1 - noise_strength])
    ax[0, 0].grid(True)

    ax[1, 1].plot(t, s2, color = plot_config['color_s2'])
    ax[1, 1].set_xticks([])
    ax[1, 1].set_yticks([])
    ax[1, 1].set_xlim(t[0], t[-1])
    # ax[1, 1].set_ylim([1 + noise_strength, - 1 - noise_strength])
    ax[1, 1].grid(True)

    # Hide the empty subplots
    ax[1, 0].set_visible(False)

    # Make figure background transparent
    fig.patch.set_alpha(0)

    return fig, ax

def save_fig(fig : plt.Figure, path_save : str, filename : str, extension_list : list) :
    """
    Save the figure in the specified path with the specified filename and extensions.
    """

    os.makedirs(os.path.dirname(path_save), exist_ok = True)

    for ext in extension_list :
        path_full = os.path.join(path_save, f"{filename}.{ext}")
        fig.savefig(path_full, bbox_inches = 'tight')

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Create the time series

t = np.linspace(0, 1, n_samples)
s1 = np.sin(2 * np.pi * f1 * t)
s2 = np.sin(2 * np.pi * f2 * t)

if add_noise :
    s1 += np.random.normal(0, noise_strength, size = s1.shape)
    s2 += np.random.normal(0, noise_strength, size = s2.shape)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Compute standard DTW

a = dtw.dtw(s1, s2, keep_internals = True)

dtw_path_to_plot = np.zeros((n_samples, n_samples))

if len(a.index1) != len(a.index2) : raise ValueError("The length of the path is different for the two time series")

for i in range(len(a.index1)) :
    idx1 = a.index1[i]
    idx2 = a.index2[i]
    dtw_path_to_plot[idx1, idx2] = 1

# Insert the value 2 in the last cell to have the same color as the other algorithms
dtw_path_to_plot[n_samples - 1, n_samples - 1] = 2

fig_dtw, ax_dtw = plot_dtw_matrix(dtw_path_to_plot, s1, s2, t, plot_config)
fig_dtw.tight_layout()
fig_dtw.show()

save_fig(fig_dtw, plot_config['path_save'], 'dtw_path', ['png', 'eps', 'pdf'])

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Create BlockDTW path and plot it

block_dtw_path_to_plot = np.ones((n_samples, n_samples)) * 2 # Initialize all the cells with the value 2 (not possible to go)

n_blocks = n_samples // block_size

for i in range(n_blocks) :
    start_idx = i * block_size
    end_idx = (i + 1) * block_size
    
    # Set the cells in the block to 1 (possible to go)
    block_dtw_path_to_plot[start_idx:end_idx, start_idx:end_idx] = 0

    # Compute the path for the block
    s1_block = s1[start_idx:end_idx]
    s2_block = s2[start_idx:end_idx]

    a_block = dtw.dtw(s1_block, s2_block, keep_internals = True)
    
    for j in range(len(a_block.index1)) :
        idx1 = a_block.index1[j] + start_idx
        idx2 = a_block.index2[j] + start_idx
        block_dtw_path_to_plot[idx1, idx2] = 1

fig_block_dtw, ax_block_dtw = plot_dtw_matrix(block_dtw_path_to_plot, s1, s2, t, plot_config)
fig_block_dtw.tight_layout()
fig_block_dtw.show()

save_fig(fig_block_dtw, plot_config['path_save'], 'block_dtw_path', ['png', 'eps', 'pdf'])

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Create overlap matrix and plot it

dtw_path_to_plot[n_samples - 1, n_samples - 1] = 1 # Reset the last cell to 1 (path goes through)
block_dtw_path_to_plot[block_dtw_path_to_plot == 2] = 0 # Set all the cells withou path to 0 ()

overlap_matrix = np.zeros((n_samples, n_samples))
for i in range(n_samples) :
    for j in range(n_samples) :
        if dtw_path_to_plot[i, j] == 1 and block_dtw_path_to_plot[i, j] == 1 :
            overlap_matrix[i, j] = 3 # Cell part of both paths (green)
        elif dtw_path_to_plot[i, j] == 1 and block_dtw_path_to_plot[i, j] == 0 :
            overlap_matrix[i, j] = 1 # Cell part of first path (black)
        elif dtw_path_to_plot[i, j] == 0 and block_dtw_path_to_plot[i, j] == 1 :
            overlap_matrix[i, j] = 2 # Cell part of second path (red)
        else :
            overlap_matrix[i, j] = 0 # Cell not part of any path (white)

fig_overlap, ax_overlap = plot_dtw_matrix(overlap_matrix, s1, s2, t, plot_config, matrix_overlap = True)
fig_overlap.tight_layout()
fig_overlap.show()

save_fig(fig_overlap, plot_config['path_save'], 'overlap_path', ['png', 'eps', 'pdf'])
