"""
Function to download and modify the data
"""
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Imports

from tslearn.datasets import CachedDatasets
import numpy as np
import matplotlib.pyplot as plt

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
