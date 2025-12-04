"""
Plot the results obtained with comparison_avg_error_prediction_computation_V2.py
The main difference with respect to comparison_avg_error_prediction_plot_V2.py is that here we plot the histogram of all the scores obtained instead of the average score per dataset.

Note that the data plotted here are the scores R defined as R = signal_length / length_dtw_path for each signal of each dataset.
This score has values between 0.5 (worst case, when length_dtw_path = 2 * signal_length) and 1 (best case, when length_dtw_path = signal_length).

@author: Alberto Zancanaro (Jesus)
@organization: Luxembourg Centre for Systems Biomedicine (LCSB)
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import seaborn as sns

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Settings

plot_config = dict(
    # Info about the data used for training and prediction
    use_z_score_normalization = True,    # If True a z-score normalization will be applied signal by signal within each dataset
    portion_of_signals_for_input = 0.85, # Portion of the signals to use for training (the rest will be used for prediction)
    n_samples_to_predict = 100,            # Number of samples to predict (If negative it is ignored and the portion_of_signals_for_input is used to define the number of samples to predict. Otherwise, this parameter override portion_of_signals_for_input)
    epoch = -1,
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    dataset_to_plot = [], # If empty plot all the datasets available together. Otherwise, specify a list of dataset names to plot separately
    create_a_separate_plot_per_dataset = False, # If True create a separate plot for each dataset specified in dataset_to_plot. 
    rescale_score_in_0_1_range = False, # If True rescale the scores of each loss function in the range [0, 1] before plotting. The rescaling is done with the formula R_rescaled = 2 * R - 1, where R is the original score.
    loss_function_to_plot = ['SDTW', 'SDTW_divergence', 'pruned_SDTW', 'OTW', 'block_SDTW_10', 'block_SDTW_50'],
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    figsize = (18, 12),
    fontsize = 16,
    aspect = 'auto',
    n_bins = 75,                           # Number of bins to use for the histogram
    normalize_hist = False,                 # If True normalize the histogram
    add_title = True,                      # If True add a title to the plot
    show_fig = True,                     # If True show the figure
    save_fig = False,                     # If True save the figure
)

loss_function_to_plot = ['MSE', 'SDTW', 'SDTW_divergence', 'pruned_SDTW', 'OTW', 'block_SDTW_10', 'block_SDTW_50']
loss_function_to_plot = ['SDTW', 'SDTW_divergence', 'pruned_SDTW', 'OTW', 'block_SDTW_10', 'block_SDTW_50']

loss_function_to_idx = dict(
    MSE = 0,
    SDTW = 1,
    SDTW_divergence = 2,
    pruned_SDTW = 3,
    OTW = 4,
    block_SDTW_10 = 5,
    block_SDTW_50 = 6
)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

plt.rcParams.update({'font.size': plot_config['fontsize']})

def create_hist(score_lists_per_loss_function : dict, ax : plt.Axes, plot_config : dict)  -> plt.Axes :
    
    # Plot the histogram of the scores for each loss function
    for loss_function in loss_function_to_plot :
        if loss_function not in plot_config['loss_function_to_plot'] : continue

        # Get the scores for the current loss function
        scores = np.array(score_lists_per_loss_function[loss_function])
        # print(len(scores), "-", loss_function, plot_config['dataset_name'])
        if len(scores) == 0 : continue
        
        # (OPTIONAL) Rescale the scores in the range [0, 1]
        if plot_config['rescale_score_in_0_1_range'] : scores = 2 * scores - 1

        # Compute the histogram
        # bins_range = (0, 1) if plot_config['rescale_score_in_0_1_range'] else (0.5, 1)
        # bins = np.linspace(bins_range[0], bins_range[1], plot_config['n_bins'] + 1)
        # histogram, _ = np.histogram(scores, bins = bins)
        #
        # # Normalize the histogram if required
        # if plot_config['normalize_hist'] : histogram = histogram / np.sum(histogram)
        # print(histogram, np.sum(histogram), loss_function, plot_config['dataset_name'])
        #
        # # Plot the histogram
        # width = np.diff(bins)
        #
        # # Plot hist
        # ax.bar(bins[:-1], histogram,
        #        width = width, align = 'edge',
        #        label = f"{loss_function} - {plot_config['dataset_name']}",
        #        )

        # Plot hist using density plot
        ax.hist(scores, 
                bins = plot_config['n_bins'],
                range = (0, 1) if plot_config['rescale_score_in_0_1_range'] else (0.5, 1),
                density = plot_config['normalize_hist'],
                alpha = 0.5,
                label = f"{loss_function} - {plot_config['dataset_name']}",
                )
    
    # Configure the plot
    ax.legend()
    ax.set_xlim([0, 1] if plot_config['rescale_score_in_0_1_range'] else [0.5, 1])
    ax.set_xlabel('Rescaled Score R' if plot_config['rescale_score_in_0_1_range'] else 'Score R', fontsize = plot_config['fontsize'])
    ax.set_ylabel('Density' if plot_config['normalize_hist'] else 'Count', fontsize = plot_config['fontsize'])
    ax.grid(True)
    if plot_config['add_title'] : ax.set_title(f"{plot_config['dataset_name']}", fontsize = plot_config['fontsize'])

    return ax

def save_fig(fig : plt.Figure, plot_config : dict, path_data : str) :
    pass

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Load data

if plot_config['n_samples_to_predict'] > 0 :
    folder_name = f"neurons_256_predict_samples_{plot_config['n_samples_to_predict']}"
else :
    folder_name = f"neurons_256_predict_portion_{int(plot_config['portion_of_signals_for_input'] * 100)}"

if plot_config['use_z_score_normalization'] : folder_name += '_z_score'

path_data = f"./saved_model/{folder_name}/0_comparison/"

if plot_config['epoch'] == -1 : plot_config['epoch'] = 'END'

with open(f"{path_data}score_lists_train_{plot_config['epoch']}.pkl", "rb") as f : score_lists_train = pickle.load(f)
with open(f"{path_data}score_lists_test_{plot_config['epoch']}.pkl", "rb") as f  : score_lists_test = pickle.load(f)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# plot_config['dataset_to_plot'] = list(score_lists_train.keys())[0:3]

if len(plot_config['dataset_to_plot']) == 0 : plot_config['dataset_to_plot'] = list(score_lists_train.keys())

for i in range(len(plot_config['dataset_to_plot'])) :
    # Get the name for the current dataset
    dataset_name = plot_config['dataset_to_plot'][i]

    # Get the score lists for the current dataset
    score_lists_dataset_train = score_lists_train[dataset_name]
    score_lists_dataset_test = score_lists_test[dataset_name]
    
    # Skip the dataset if there are no scores
    # Note that if there are no scores for the train set, there will be no scores for the test set
    if len(score_lists_dataset_train) == 0 :
        if not plot_config['create_a_separate_plot_per_dataset'] and i == len(plot_config['dataset_to_plot']) - 1 :
            pass
        else :
            continue

    if plot_config['create_a_separate_plot_per_dataset'] :

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Plot the histogram for the current dataset - Train

        fig, ax = plt.subplots(figsize = plot_config['figsize'])

        plot_config['dataset_name'] = f"{dataset_name} - Train"
        ax = create_hist(score_lists_dataset_test, ax, plot_config)
        fig.tight_layout()

        # (OPTIONAL) show the figure
        if plot_config['show_fig'] : plt.show()

        # (OPTIONAL) Save the figure
        if plot_config['save_fig'] : save_fig(fig, plot_config, path_data)

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Plot the histogram for the current dataset - Test
    
        fig, ax = plt.subplots(figsize = plot_config['figsize'])

        plot_config['dataset_name'] = f"{dataset_name} - Test"
        ax = create_hist(score_lists_dataset_train, ax, plot_config)
        fig.tight_layout()

        # (OPTIONAL) show the figure
        if plot_config['show_fig'] : plt.show()

        # (OPTIONAL) Save the figure
        if plot_config['save_fig'] : save_fig(fig, plot_config, path_data)

    else :
        # Merge the score lists of all datasets
        print(i)
        
        # Initialize the merged score lists at the first iteration
        if i == 0 :
            merged_score_lists_train = dict()
            merged_score_lists_test = dict()
            for loss_function in loss_function_to_plot :
                merged_score_lists_train[loss_function] = []
                merged_score_lists_test[loss_function] = []
        
        # Merge the score lists
        for loss_function in loss_function_to_plot :
            merged_score_lists_train[loss_function] += score_lists_dataset_train[loss_function] if loss_function in score_lists_dataset_train else []
            merged_score_lists_test[loss_function] += score_lists_dataset_test[loss_function] if loss_function in score_lists_dataset_test else []
        
        # Plot the histogram only at the last iteration
        if i == len(plot_config['dataset_to_plot']) - 1 :
            # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            # Plot - Train

            fig, ax = plt.subplots(figsize = plot_config['figsize'])

            plot_config['dataset_name'] = "All Datasets - Test"
            ax = create_hist(merged_score_lists_train, ax, plot_config)
            fig.tight_layout()

            # (OPTIONAL) show the figure
            if plot_config['show_fig'] : plt.show()

            # (OPTIONAL) Save the figure
            if plot_config['save_fig'] : save_fig(fig, plot_config, path_data)

            # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            # Plot - Test

            fig, ax = plt.subplots(figsize = plot_config['figsize'])

            plot_config['dataset_name'] = "All Datasets - Test"
            ax = create_hist(merged_score_lists_test, ax, plot_config)
            fig.tight_layout()

            # (OPTIONAL) show the figure
            if plot_config['show_fig'] : plt.show()

            # (OPTIONAL) Save the figure
            if plot_config['save_fig'] : save_fig(fig, plot_config, path_data)

                
# # Example of using seaborn to plot a distribution with KDE
# sns_plot = sns.displot(data = merged_score_lists_train,
#                        # bins = plot_config['n_bins'],
#                        kind = 'kde',
#                          # kde = True,
#                        )
# plt.grid()
# plt.show()
#
#
# import statsmodels.api as sm
# from statsmodels.graphics.gofplots import qqplot_2samples
# x = np.asarray(merged_score_lists_train['SDTW']) 
# pp_x = sm.ProbPlot(x)
# for loss in merged_score_lists_train.keys():
#     if loss == 'SDTW':
#         continue
#     y = np.asarray(merged_score_lists_train[loss])
#     pp_y = sm.ProbPlot(y)
#     qqplot_2samples(pp_x, pp_y, line = '45', xlabel = 'SDTW', ylabel = loss)
# plt.title(f"QQ plot SDTW vs {loss} - Train")
# plt.grid()
# plt.show()
#         
#
#
#
#
#
#
