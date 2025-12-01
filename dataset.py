"""
Function to download and modify the data

@author: Alberto Zancanaro (Jesus)
@organization: Luxembourg Centre for Systems Biomedicine (LCSB)
"""
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

from tslearn.datasets import CachedDatasets

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def get_dataset() :
    """
    Get a toy dataset from the tslearn library
    """
    data_loader = CachedDatasets()
    x_train, y_train, x_test, y_test = data_loader.load_dataset("Trace")

    return x_train, y_train, x_test, y_test

def get_subset(x_train, y_train, n = 50) :
    """
    Get a subset of the dataset
    """
    x_subset = x_train[y_train < 4]
    np.random.shuffle(x_subset)
    
    if n > len(x_subset) :
        print(f"Warning: the subset size is larger than the dataset size. The subset size will be set to the dataset size ({len(x_subset)})")
        n = len(x_subset)

    x_subset = x_subset[:n]

    return x_subset

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def z_score_normalization(x : np.ndarray) -> np.ndarray :
    """
    Z-score normalization of the input signal

    Parameters
    ----------
    x : numpy array
        The input signal of shape (n_signal, signal_length)

    Returns
    -------
    x_normalized : numpy array
        The normalized signal of shape (n_signal, signal_length)
    """
    x_mean = np.mean(x, axis = 1, keepdims = True)
    x_std  = np.std(x, axis = 1, keepdims = True)

    x_normalized = (x - x_mean) / x_std

    return x_normalized

def generate_signals(x, n_signals_to_generate : int, length_1 : int = 150, length_2 : int = 150) :
    """
    From the input signal 'x', generate 'n_signal' pairs.
    To generate the pair a random point is selected and the signal is divided in two parts.
    The first part has length 'length_1' and the second part has length 'length_2'
    Works only if the length of the input signal is larger than 'length_1' + 'length_2'
    
    Parameters
    ----------
    x : numpy array
        The input signal of shape (n_signal, signal_length, 1)
    n_signals_to_generate : int
        The number of signals to generate_signal
    length_1 : int
        The length of the first part of the signal
    length_2 : int
        The length of the second part of the signal

    Returns
    -------
    signals_1 : numpy array
        Array of shape (n_signals_to_generate, length_1, 1) containing the first part of the signals
    signals_2 : numpy array
        Array of shape (n_signals_to_generate, length_2, 1) containing the second part of the signals
    """
    
    # Check if the input signal is long enough
    if x.shape[1] <= length_1 + length_2 : raise ValueError("The length of the input signal is smaller than the specified length")
    
    # Variables to used during the generation
    signals_1 = []
    signals_2 = []
    signal_original = []
    t_original = [] # Save the starting point of signal 1 in the original signal
    i = 0

    while len(signals_1) < n_signals_to_generate  :
        tmp_signal = x[i]

        while True :
            # Generate the random point
            random_point = np.random.randint(0, len(tmp_signal) - length_2)
            
            # Check if the random point is valid
            if random_point - length_1 < 0 or random_point + length_2 > len(tmp_signal) :
                continue

            # Break the loop
            break

        # Get the two parts of the signal
        signal_1 = tmp_signal[random_point - length_1 : random_point]
        signal_2 = tmp_signal[random_point : random_point + length_2]

        # Append the pair to the list
        signals_1.append(signal_1)
        signals_2.append(signal_2)
        signal_original.append(tmp_signal)
        t_original.append(random_point - length_1)

        i += 1
        if i >= len(x) : i = 0

    return np.array(signals_1), np.array(signals_2), np.array(signal_original), np.array(t_original)

def generate_signals_V2(x : np.ndarray, point_for_division : int) :
    """
    Given a matrix of shape (n_signals, signal_length), generate two parts for each signal (i.e. each row of the matrix).
    The first part is from the beginning of the signal to the point specified in the parameter `point_for_division`.
    The second part is from the point specified in the parameter `point_for_division` to the end of the signal.
    The function returns two matrices, one for each part of the signal.
    If point_for_division is larger than the length of the signal or smaller than 0, an error is raised.

    Parameters
    ----------
    x : numpy array
        The input signal of shape (n_signals, signal_length)
    point_for_division : int
        The point where the signal is divided. It should be between 0 and the length of the signal.
    """

    if point_for_division <= 0 or point_for_division > x.shape[1] :
        raise ValueError(f"point_for_division must be between 0 and the length of the signal (inclusive) ({x.shape[1]}). Current value is {point_for_division}")

    # Get the first part of the signal
    signals_1 = x[:, :point_for_division]

    # Get the second part of the signal
    signals_2 = x[:, point_for_division:]

    return signals_1, signals_2

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def visualize_signals(x_1, x_2, x_original = None, t_orig = None, visualize_plot = True) :
    if t_orig is None and x_original is not None : raise ValueError("If the original signal is provided, the starting point of the first signal must be provided")
    if t_orig is not None and x_original is None : raise ValueError("If the starting point of the first signal is provided, the original signal must be provided")

    fig, ax = plt.subplots(1, 1, figsize = (10, 10))

    if t_orig is not None : shift = t_orig
    else : shift = 0
    t_1  = np.arange(0, len(x_1)) + shift
    t_2  = np.arange(len(x_1), len(x_1) + len(x_2)) + shift

    ax.plot(t_1, x_1, label = 'First part of the signal')
    ax.plot(t_2, x_2, label = 'Second part of the signal')
    if x_original is not None : ax.plot(np.arange(0, len(x_original)), x_original - np.random.rand(), label = 'Original signal')

    ax.grid()
    ax.legend()
    ax.set_xlim(0, len(x_1) + len(x_2))
    ax.set_xlabel('Time')

    fig.tight_layout()
    if visualize_plot : fig.show()

    return fig, ax


def visualize_prediction(ts_index, x_orig, start_prediction,
                         x_pred_list : list, x_pred_label : list[str],
                         show_figure = True, path_save : str = None, plot_title : str = None) :
    """
    Function to visualize the prediction of the time series.

    Parameters
    ----------
    ts_index : int
        The index of the time series to visualize.
    x_orig : numpy array
        The original time series data of shape (n_signals, signal_length).
    start_prediction : int
        The starting point of the prediction in the time series.
    x_pred_list : list
        A list containing the predicted time series data. Each element of the list should be a numpy array of shape (n_signals, signal_length).
    x_pred_label : list of str
        A list containing the labels for each predicted time series in `x_pred_list`. The length of this list should be the same as the length of `x_pred_list`.
    show_figure : bool, optional
        If True, the figure will be shown. Default is True.
    path_save : str, optional
        If provided, the figure will be saved to this path. If None, the figure will not be saved. Default is None.
        The path should include the filename and the extension (e.g. 'path/to/save/figure.png').
        If the folder does not exist, it will be created.
    plot_title : str, optional
        The title of the plot. If None, no title will be set. Default is None
    """
    
    # Remove the last dimension if it is 1
    x_orig = x_orig.squeeze()  # Ensure x_orig is 2D

    # Check input data
    if x_orig.ndim != 2 : raise ValueError("x_orig must be a 2D array of shape (n_signals, signal_length)")
    if ts_index < 0 or ts_index >= x_orig.shape[0] : raise ValueError(f"ts_index must be between 0 and {x_orig.shape[0] - 1}. Current value is {ts_index}")

    fig, ax = plt.subplots(1, 1, figsize = (12, 9))
    fontsize = 22
    linewidth = 1.3
    ax.plot(x_orig[ts_index].ravel(), label = 'Original Signal', color = 'black')

    color_map = dict(
        MSE = 'g',
        SDTW = 'tab:red',
        SDTW_divergence = 'brown',
        pruned_SDTW = 'violet',
        block_SDTW = 'blue',
        OTW = 'orange',
    )

    for i in range(len(x_pred_list)) :
        x_pred = x_pred_list[i].squeeze()  # Ensure each x_pred is 2D
        label = x_pred_label[i]

        len_prediction = len(x_pred[ts_index])
    
        t_prediction = np.arange(start_prediction, start_prediction + len_prediction)
        ax.plot(t_prediction, x_pred[ts_index],
                linewidth = linewidth,
                label = label, color = color_map[label]
                )
        
    if plot_title is not None : ax.set_title(plot_title)
    ax.set_ylabel('Amplitude [A.U.]', fontsize = fontsize)
    ax.set_xlabel('Time samples', fontsize = fontsize)
    ax.tick_params(axis = 'both', which = 'major', labelsize = fontsize)

    ax.grid()
    ax.legend(fontsize = fontsize)

    fig.tight_layout()
    if show_figure : fig.show()
    
    print(f"Visualizing time series {ts_index} with starting point {start_prediction} for prediction")
    print(path_save)
    if path_save is not None :
        folder_path = os.path.dirname(path_save)
        if not os.path.exists(folder_path) : os.makedirs(folder_path)

        fig.savefig(path_save)
        print(f"Figure saved to {path_save}")

    return fig, ax

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def read_UCR_dataset(path_folder_dataset : str, name_dataset : str) -> tuple :
    """
    Function to read and use the data from the UCR Time Series Classification Archive.
    See here for more information : https://www.cs.ucr.edu/%7Eeamonn/time_series_data_2018/

    Parameters
    ----------
    path_folder_dataset : str
        The path to the folder containing the dataset. The folder should contain the files named as xxx_TRAIN.tsv and xxx_TEST.tsv, where xxx is the name of the dataset specified in the parameter `name_dataset`.
    name_dataset : str
        The name of the dataset to read. It should be the name of the dataset without the '_train' or '_test' suffix.

    Returns
    -------
    data_train : numpy array
        The training data from the dataset, excluding the first column (labels).
    labels_train : numpy array
        The labels from the training data, which are in the first column.
    data_test : numpy array
        The test data from the dataset, excluding the first column (labels).
    labels_test : numpy array
        The labels from the test data, which are in the first column.
    """

    path_train = os.path.join(path_folder_dataset, name_dataset + '_TRAIN.tsv')
    data_train, labels_train = read_tsv_file(path_train)

    path_test = os.path.join(path_folder_dataset, name_dataset + '_TEST.tsv')
    data_test, labels_test = read_tsv_file(path_test)

    return data_train, labels_train, data_test, labels_test


def read_tsv_file(path_tsv_file : str) -> tuple :
    """
    Function to read one of the tsv files from the UCR Time Series Classification Archive.

    Parameters
    ----------
    path_tsv_file : str
        The path to the tsv file. The file should be in the format 'name_of_the_dataset_train.tsv' or 'name_of_the_dataset_test.tsv'.

    Returns
    -------
    data : numpy array
        The data from the tsv file, excluding the first column (labels).
    labels : numpy array
        The labels from the tsv file, which are in the first column.
    """
    
    # Read file and convert to numpy
    tmp_df = pd.read_csv(filepath_or_buffer = path_tsv_file, delimiter = '\t', quotechar = '"')
    tmp_data = tmp_df.to_numpy()
    
    # Get data and labels
    data = tmp_data[:, 1:]
    labels = tmp_data[:, 0]

    return data, labels
