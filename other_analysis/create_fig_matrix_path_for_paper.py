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

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Settings

map_size = 300

diagonal_size = 75  # Size of the diagonal band for PrunedDTW
block_size = 75

plot_config = dict(
    figsize = (20, 20),
    aspect = 'equal',
    path_save = './other_analysis/figures/'
)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Path simulations

def create_path_dtw(dtw_map : np.ndarray) -> np.ndarray :
    """
    Create a hypothetical DTW path from (0, 0) to (map_size-1, map_size-1).
    The path is created by randomly choosing to move either down or right at each step, until reaching the bottom-right corner.
    The function receives as input the DTW map where the path will be created.
    The DTW map could have 2 possible values : 
    - 0 : cell where it is possible to go (in the case of the standard DTW all the cells are possible. For PrunedDTW and BlockDTW some cells will be precluded)
    - 2 : cell where it is not possible to go (only for PrunedDTW and BlockDTW)

    The function returns the map updated with the path, saved as cells with value 1.

    Parameters
    ----------
    dtw_map : np.ndarray
        The DTW map where the path will be created.

    Returns
    -------
    np.ndarray
        The DTW map updated with the path.
    """

    i, j = 0, 0

    while (i < dtw_map.shape[0] - 1) or (j < dtw_map.shape[1] - 1) :
        dtw_map[i, j] = 1  # Mark the cell as part of the path

        if i == dtw_map.shape[0] - 1 or dtw_map[i + 1, j] == 2 :
            # It can only move right because it has reached the last row or the cell below is precluded
            j += 1  
        elif j == dtw_map.shape[1] - 1 or dtw_map[i, j + 1] == 2 :
            # It can only move down because it has reached the last column or the cell to the right is precluded
            i += 1
        else :
            # It can move either down or right
            if np.random.rand() < 0.5 :
                i += 1  # Move down
            else :
                j += 1  # Move right

    return dtw_map


def create_mask_pruned_dtw(map_size : int, diagonal_size : int) :
    """
    Create a mask for the PrunedDTW algorithm. The mask will have 1 in the cells that are possible to go and 0 in the cells that are not possible to go.
    The mask will be centered around the diagonal with a given diagonal size.
    """

    mask = np.ones((map_size, map_size)) * 2

    for i in range(map_size) :
        for j in range(map_size) :
            if abs(i - j) <= diagonal_size :
                mask[i, j] = 0

    return mask

def create_mask_block_dtw(map_size : int, block_size : int) :
    """
    Create a mask for the BlockDTW algorithm. The mask will have 1 in the cells that are possible to go and 0 in the cells that are not possible to go.
    The mask will be created by dividing the matrix into blocks of given block size and allowing only the blocks along the diagonal.
    """

    mask = np.ones((map_size, map_size)) * 2

    num_blocks = map_size // block_size

    for b in range(num_blocks) :
        start = b * block_size
        end = start + block_size
        mask[start:end, start:end] = 0

    return mask

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def plot_dtw_matrix(dtw_matrix : np.ndarray, plot_config : dict) :
    """
    Plot the DTW matrix. The DTW matrix has the following values :
    - 0 : cell where it is possible to go
    - 1 : cell where the path goes through
    - 2 : cell where it is not possible to go

    The cells are colored as follows :
    - 0 : light red
    - 1 : black
    - 2 : white
    """

    fig, ax = plt.subplots(figsize = plot_config['figsize'])

    # Create a colormap
    cmap_colors = np.zeros((3, 4)) # 3 colors, 4 channels (RGBA)
    cmap_colors[0] = [1, 0.8, 0.8, 1]  # Light red for cell where it is possible to go (0)
    cmap_colors[1] = [0, 0, 0, 1]   # Black for cell where the path goes through (1)
    cmap_colors[2] = [1, 1, 1, 1]      # White for cell where it is not possible to go (2)
    custom_cmap = plt.matplotlib.colors.ListedColormap(cmap_colors)

    ax.imshow(dtw_matrix.T, cmap = custom_cmap, aspect = plot_config['aspect'])

    ax.set_xticks([])
    ax.set_yticks([])
    # ax.set_xticklabels([], minor = False)
    # ax.set_yticklabels([], minor = False)
    # ax.grid(which = 'major', color = 'k', linestyle = '-', snap = False)

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

# Create DTW path and plot it
dtw_matrix = np.zeros((map_size, map_size))
dtw_matrix = create_path_dtw(dtw_matrix)
dtw_matrix[map_size - 1, map_size - 1] = 2 # Insert the value 2 in the last cell to have the same color as the other algorithms

fig_dtw, ax_dtw = plot_dtw_matrix(dtw_matrix, plot_config)
fig_dtw.tight_layout()
fig_dtw.show()

save_fig(fig_dtw, plot_config['path_save'], 'dtw_path', ['png', 'eps', 'pdf'])

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Create PrunedDTW path and plot it
pruned_dtw_mask = create_mask_pruned_dtw(map_size, diagonal_size)
pruned_dtw_matrix = create_path_dtw(pruned_dtw_mask)

fig_pruned_dtw, ax_pruned_dtw = plot_dtw_matrix(pruned_dtw_matrix, plot_config)
fig_pruned_dtw.tight_layout()
fig_pruned_dtw.show()

save_fig(fig_pruned_dtw, plot_config['path_save'], 'pruned_dtw_path', ['png', 'eps', 'pdf'])

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Create BlockDTW path and plot it
block_dtw_mask = create_mask_block_dtw(map_size, block_size)
block_dtw_matrix = create_path_dtw(block_dtw_mask)

fig_block_dtw, ax_block_dtw = plot_dtw_matrix(block_dtw_matrix, plot_config)
fig_block_dtw.tight_layout()
fig_block_dtw.show()

save_fig(fig_block_dtw, plot_config['path_save'], 'block_dtw_path', ['png', 'eps', 'pdf'])

# plt.close('all')
