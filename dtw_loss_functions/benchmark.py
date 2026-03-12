"""
Module for benchmarking the DTW loss functions implemented in this package. It includes functions to compare the performance of different DTW loss, save and visualize the results.

Authors
-------
Alberto Zancanaro <alberto.zancanaro@uni.lu>

"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

import numpy as np
import matplotlib.pyplot as plt
import os
import time
import torch

from dtw_loss_functions import block_dtw, soft_dtw, otw

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def compute_benchmark(B_list : list, T_list : list, C_list : list, loss_functions_to_use : dict, device : str, n_repetitions : int = 100, path_save : str = 'benchmark', print_progress : bool = False) :
    """
    For each combination of batch size, signal length and number of channels specified in the input lists, compute the loss for each loss function specified in ``loss_functions_to_use``.
    
    The ``loss_functions_to_use`` is a dictionary where the keys are the names of the loss functions and the values are the instances of the loss functions to test.
    The loss functions must be instances of the classes implemented in this package.

    The results are saved in npy files at the path ``{path_save}/{loss_function_name}/``. The ``loss_function_name`` is the key of the loss function in the ``loss_functions_to_use`` dictionary.
    Each file will be name as ``B_{batch_size}_T_{signal_length}_C_{num_channels}.npy`` and will contain a numpy array of shape (n_repetitions,) with the computed loss values for each repetition.
    
    Parameters
    ----------
    B_list : list
        List of batch sizes to test.
    T_list : list
        List of signal lengths to test.
    C_list : list
        List of number of channels to test.
    loss_functions_to_use : dict
        Dictionary of loss functions to test. The keys are the names of the loss functions and the values are the instances of the loss functions to test.
    device : str
        Device to use for the computations. Must be 'cpu' or 'cuda'.
    n_repetitions : int, optional
        Number of repetitions to perform for each combination of batch size, signal length and number of channels. The default value is 100.
    path_save : str, optional
        Path to save the results. The default value is 'benchmark'. If the folder does not exist, it will be created. If results are already present in the folder, they will be overwritten.
    print_progress : bool, optional
        Whether to print the progress of the benchmark. The default value is False.
    """
    
    # Create the path if it does not exist
    os.makedirs(path_save, exist_ok = True)
    
    # Iterate over loss functions
    for loss_function_name in loss_functions_to_use :
        if print_progress : print(f"Loss function : {loss_function_name}")
        current_loss_function = loss_functions_to_use[loss_function_name]

        # Create folder for the current loss function if it does not exist
        path_save_loss_function = os.path.join(path_save, loss_function_name)
        os.makedirs(path_save_loss_function, exist_ok = True)
        
        # Iterate over batch sizes
        for i in range(len(B_list)) :
            if print_progress : print(f"\tBatch size = {B_list[i]} ({i + 1} / {len(B_list)})")
            batch_size = B_list[i]
            
            # Iterate over signal lengths
            for j in range(len(T_list)) :
                if print_progress : print(f"\t\tSignal length = {T_list[j]} ({j + 1} / {len(T_list)})")
                time_samples = T_list[j]
                
                # Iterate over number of channels
                for k in range(len(C_list)) :
                    if print_progress : print(f"\t\t\tNumber of channels = {C_list[k]} ({k + 1} / {len(C_list)})")
                    n_channels = C_list[k]

                    # Create synthetic data
                    x   = torch.rand(batch_size, time_samples, n_channels).to(device)
                    x_r = torch.rand(batch_size, time_samples, n_channels).to(device)

                    # Evaluate computation time (over n_repetitions)
                    time_list_loss = repeat_inference(x, x_r, current_loss_function, n_repetitions)

                    # Save results
                    file_name = f"B_{batch_size}_T_{time_samples}_C_{n_channels}.npy"
                    path_save_file = os.path.join(path_save_loss_function, file_name)
                    np.save(path_save_file, time_list_loss)

def repeat_inference(x, x_r, loss_function : torch.nn.Module, n_repetitions : int = 100) -> list :
    """
    Compute the loss for the input tensors ``x`` and ``x_r`` for a number of repetitions specified in ``n_repetitions`` and save the computation times in a list.

    Parameters
    ----------
    x : torch.tensor
        First input tensor of shape ``B x T x C``.
    x_r : torch.tensor
        Second input tensor of shape ``B x T x C``.
    loss_function : torch.nn.Module
        Loss function to use for the computation. The loss function must be an instance of a class implemented in this package.
    n_repetitions : int
        Number of repetitions to perform. The default value is ``100``.

    Returns
    -------
    time_list_loss : list
        List of length ``n_repetitions`` containing the computation times for each repetition.
    """
    time_list_loss = np.zeros(n_repetitions)

    with torch.no_grad() :
        for i in range(n_repetitions) :
            
            # Evaluate computation time
            time_start = time.time()
            _ = loss_function(x, x_r)
            time_end = time.time()

            # Save computations times
            time_list_loss[i] = time_end - time_start

    return time_list_loss

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def load_benchmark_results(path_results : str, loss_list : list = None) -> dict :
    """
    Load all the results stored in the specified path. This function expects that the results are stored in npy files organized as described in the docstring of the :func:``compute_benchmark`` function.
    The function will return a dictionary where the keys are the loss function names and the values are dictionaries containing the results for each combination of batch size, signal length and number of channels.

    Note that by default, the function will load all the results present in the specified path. The name of the loss function will be inferred from the filename.
    If you want to load only a subset of the results, you can specify the list of loss function names to load in the ``loss_list`` parameter. In this case, only the results corresponding to the specified loss function names will be loaded.

    Parameters
    ----------
    path_results : str
        Path to the folder containing the results. The results must be stored in npy files with the name format ``B_{batch_size}_T_{signal_length}_C_{num_channels}.npy``.
    loss_list : list, optional
        List of loss function names to load. If specified, only the results corresponding to the specified loss function names will be loaded. The default value is None, which means that all the results present in the specified path will be loaded.

    Returns
    -------
    results_dict : dict[str, dict[str, np.ndarray]]
        Dictionary containing the loaded results. The keys are the loss function names and the values are dictionaries containing the results for each combination of batch size, signal length and number of channels.
    """
    
    # Variable to store the results
    results_dict = {}

    # Get all folders in the specified path.
    # Note that the function compute benchmark should create a folder for each loss function (with the folder name corresponding to the keys of dictionary that contains the loss functions)
    # This list iteration first will get all element inside path results (f for f in os.listdir(path_results)). Then that element will be inserted in the list only if it is a folder (if os.path.isdir(os.path.join(path_results, f))).
    folder_list = [f for f in os.listdir(path_results) if os.path.isdir(os.path.join(path_results, f))]
    
    # Iterate over folders (i.e. loss functions)
    for folder in folder_list :
        # Get the name of the loss function from the folder name
        loss_function_name = folder

        # Check if the loss function name is in the list of loss functions to load (if specified)
        if loss_list is not None :
            if loss_function_name not in loss_list :
                continue

        # Create a dictionary to store the results for the current loss function
        results_dict[loss_function_name] = {}

        # Get all npy files in the current folder
        path_folder = os.path.join(path_results, folder)
        file_list = [f for f in os.listdir(path_folder) if f.endswith('.npy')]

        # Check if there are npy files in the current folder
        if len(file_list) == 0 :
            print(f"Warning : no npy files found in folder {path_folder}. Skipping this folder.")
            continue
        
        # Iterate over file for the specific loss function
        for filename in file_list :
            # batch_size, signal_length, n_channels = get_info_from_filename(filename)

            # Remove the extension from the filename
            filename_no_ext = filename.split('.')[0]

            # Save the results in the dictionary
            # Note that the key for each result was chosen to be the filename without the extension because it is very easy to create.
            results_dict[loss_function_name][filename_no_ext] = np.load(os.path.join(path_folder, filename))

    return results_dict

def get_array_to_plot(results_dict : dict, loss_function_name : str, x_axis_variable : str, B_list : list, T_list : list, C_list : list) -> tuple :
    """
    Given the results dictionary returned by the :func:``load_benchmark_results`` function, extract the array of computation times for the specified loss function name.
    The data will be organized as a list of arrays, where each array corresponds to a specific combination of batch size, signal length and number of channels. 
    The x-axis variable indicates which variable should be used as x-axis in the plot.
    E.g. if x_axis_variable is 'T', the function will return a list of arrays where each array corresponds to a specific combination of batch size and number of channels, and the values in the array correspond to the computation times for different signal lengths.

    
    This function will return 3 lists :

    - data_list_mean : list of arrays containing the mean computation times for each combination of batch size, signal length and number of channels.
    - data_list_std : list of arrays containing the standard deviation of the computation times for each combination of batch size, signal length and number of channels.
    - data_list_label : list of strings containing, where each element corresponds to the label of the data in the same position in the data_list_mean and data_list_std lists.

    Remember that for each combination of batch size, signal length and number of channels, there are n_repetitions computation times stored in the results dictionary. 
    For this reason, this function return both the mean and std for each combination.

    Parameters
    ----------
    results_dict : dict
        Dictionary containing the loaded results as returned by the :func:``load_benchmark_results`` function.
    loss_function_name : str
        Name of the loss function for which to extract the data. It must be a key of the input dictionary.
    x_axis_variable : str
        Variable to use as x-axis in the plot. It must be one of 'B', 'T' or 'C', corresponding to batch size, signal length and number of channels, respectively.
    """

    if loss_function_name not in results_dict :
        raise ValueError(f"Loss function name {loss_function_name} not found in the results dictionary. Available loss function names are : {list(results_dict.keys())}")

    if x_axis_variable not in ['B', 'T', 'C'] :
        raise ValueError(f"x_axis_variable must be one of 'B', 'T' or 'C'. Received {x_axis_variable}")

    if x_axis_variable == 'T' :
        data_list_mean, data_list_std, data_list_label = __get_array_to_plot_T(results_dict, loss_function_name, B_list, T_list, C_list)
    elif x_axis_variable == 'B' :
        data_list_mean, data_list_std, data_list_label = __get_array_to_plot_B(results_dict, loss_function_name, B_list, T_list, C_list)
    elif x_axis_variable == 'C' :
        data_list_mean, data_list_std, data_list_label = __get_array_to_plot_C(results_dict, loss_function_name, B_list, T_list, C_list)

    return data_list_mean, data_list_std, data_list_label

def __get_array_to_plot_T(results_dict, loss_function_name, B_list, T_list, C_list) :
    data_list_mean = []
    data_list_std = []
    data_list_label = []

    for B in B_list :
        for C in C_list :
            
            # Temporary variables to store the mean and std of the computation times for the current combination of batch size and number of channels.
            tmp_data_mean = np.zeros(len(T_list))
            tmp_data_std = np.zeros(len(T_list))
            tmp_data_label = f"{loss_function_name} (B={B}, C={C})"

            for i in range(len(T_list)) :
                T = T_list[i]

                # Create the key to access the results for the current combination of batch size, signal length and number of channels
                key = f"B_{B}_T_{T}_C_{C}"
                if key not in results_dict[loss_function_name] : raise ValueError(f"Key {key} not found in the results dictionary for loss function {loss_function_name}. Check your input lists and the results dictionary.")

                # Get the computation times for the current combination of batch size, signal length and number of channels
                time_values = results_dict[loss_function_name][key]

                # Save mean and std
                tmp_data_mean[i] = np.mean(time_values)
                tmp_data_std[i] = np.std(time_values)
            
            # Save data and label
            data_list_mean.append(tmp_data_mean)
            data_list_std.append(tmp_data_std)
            data_list_label.append(tmp_data_label)

    return data_list_mean, data_list_std, data_list_label

def __get_array_to_plot_B(results_dict, loss_function_name, B_list, T_list, C_list) :
    data_list_mean = []
    data_list_std = []
    data_list_label = []

    for T in T_list :
        for C in C_list :
            
            # Temporary variables to store the mean and std of the computation times for the current combination of signal length and number of channels.
            tmp_data_mean = np.zeros(len(B_list))
            tmp_data_std = np.zeros(len(B_list))
            tmp_data_label = f"{loss_function_name} (T={T}, C={C})"

            for i in range(len(B_list)) :
                B = B_list[i]

                # Create the key to access the results for the current combination of batch size, signal length and number of channels
                key = f"B_{B}_T_{T}_C_{C}"
                if key not in results_dict[loss_function_name] : raise ValueError(f"Key {key} not found in the results dictionary for loss function {loss_function_name}. Check your input lists and the results dictionary.")

                # Get the computation times for the current combination of batch size, signal length and number of channels
                time_values = results_dict[loss_function_name][key]

                # Save mean and std
                tmp_data_mean[i] = np.mean(time_values)
                tmp_data_std[i] = np.std(time_values)

            # Save data and label
            data_list_mean.append(tmp_data_mean)
            data_list_std.append(tmp_data_std)
            data_list_label.append(tmp_data_label)

    return data_list_mean, data_list_std, data_list_label

def __get_array_to_plot_C(results_dict, loss_function_name, B_list, T_list, C_list) :
    data_list_mean = []
    data_list_std = []
    data_list_label = []

    for B in B_list :
        for T in T_list :
            
            # Temporary variables to store the mean and std of the computation times for the current combination of batch size and signal length.
            tmp_data_mean = np.zeros(len(C_list))
            tmp_data_std = np.zeros(len(C_list))
            tmp_data_label = f"{loss_function_name} (B={B}, T={T})"

            for i in range(len(C_list)) :
                C = C_list[i]

                # Create the key to access the results for the current combination of batch size, signal length and number of channels
                key = f"B_{B}_T_{T}_C_{C}"
                if key not in results_dict[loss_function_name] : raise ValueError(f"Key {key} not found in the results dictionary for loss function {loss_function_name}. Check your input lists and the results dictionary.")

                # Get the computation times for the current combination of batch size, signal length and number of channels
                time_values = results_dict[loss_function_name][key]

                # Save mean and std
                tmp_data_mean[i] = np.mean(time_values)
                tmp_data_std[i] = np.std(time_values)

            # Save data and label
            data_list_mean.append(tmp_data_mean)
            data_list_std.append(tmp_data_std)
            data_list_label.append(tmp_data_label)

    return data_list_mean, data_list_std, data_list_label

def get_info_from_filename(file_name : str) -> tuple :
    """
    Given a file name in the format ``B_{batch_size}_T_{signal_length}_C_{num_channels}.npy``, extract the batch size, signal length and number of channels from the filename and return them.

    Parameters
    ----------
    file_name : str
        File name in the format ``B_{batch_size}_T_{signal_length}_C_{num_channels}.npy``.

    Returns
    -------
    batch_size : int
        Batch size extracted from the file name.
    signal_length : int
        Signal length extracted from the file name.
    n_channels : int
        Number of channels extracted from the file name.
    """
    
    # Remove the extension from the file name
    file_name = file_name.split('.')[0]

    # Extract the batch size, signal length and number of channels from the file name
    _, B_str, _, T_str, _, C_str = file_name.split('_')

    # Convert the extracted values to integers
    batch_size = int(B_str)
    signal_length = int(T_str)
    n_channels = int(C_str)

    # Return the extracted values
    return batch_size, signal_length, n_channels

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def plot_benchmark(plot_config : dict, x_axis_variable_list : list, x_axis_variable_name : str, data_to_plot_mean : list, labels_to_plot : list, data_to_plot_std : list = None) :
    """
    Plot the results of the benchmark. The data should be in the format returned by the :func:``get_array_to_plot`` function, i.e. a list of arrays containing the mean computation times for each combination of batch size, signal length and number of channels, a list of strings containing the labels for each array.
    The x-axis variable indicates which variable should be used as x-axis in the plot. E.g. if x_axis_variable is 'T', the function will plot the computation times for different signal lengths.
    If the standard deviation of the computation times is provided, it will be plotted as shaded area around the mean values.
    
    Parameters
    ----------
    """

    # Check plot settings
    if 'figsize' not in plot_config : plot_config['figsize'] = (10, 6)
    if 'y_scale_log' not in plot_config : plot_config['y_scale_log'] = False
    if 'use_milliseconds' not in plot_config : plot_config['use_milliseconds'] = False
    if 'fontsize' not in plot_config : plot_config['fontsize'] = 12
    
    # Create the figure and axis
    fig, ax = plt.subplots(figsize = plot_config['figsize'])

    # 
    if 'use_milliseconds' in plot_config and plot_config['use_milliseconds'] :
        scale_factor = 1000
        label_time_unit = 'ms'
    else :
        scale_factor = 1
        label_time_unit = 's'
        
    for i in range(len(data_to_plot_mean)) :
        # Plot mean values
        mean_values = data_to_plot_mean[i] * scale_factor
        label = labels_to_plot[i]
        ax.plot(x_axis_variable_list, mean_values, label = label)

        # (OPTIONAL) Plot std values as shaded area around the mean values
        if data_to_plot_std is not None :
            std_values = data_to_plot_std[i] * scale_factor
            ax.fill_between(x_axis_variable_list, mean_values - std_values, mean_values + std_values, alpha = 0.2)
    
    # Add details to the plot
    ax.set_xlabel(x_axis_variable_name, fontsize = plot_config['fontsize'])
    ax.set_ylabel(f"Computation time ({label_time_unit})", fontsize = plot_config['fontsize'])
    ax.set_xlim(x_axis_variable_list[0], x_axis_variable_list[-1])
    if plot_config['y_scale_log'] : ax.set_yscale('log')
    ax.legend(fontsize = plot_config['fontsize'])
    ax.grid(True)
    fig.tight_layout()
    ax.tick_params(axis = 'both', which = 'major', labelsize = plot_config['fontsize'] * 0.8)

    return fig, ax
