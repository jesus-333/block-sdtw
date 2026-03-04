"""
Implementation of OTW, from "OTW: Optimal Transport Warping for Time Series"

For more details see :
- https://ieeexplore.ieee.org/document/10095915
- https://arxiv.org/abs/2306.00620
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

import torch

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def otw_distance(x : torch.Tensor, y : torch.Tensor, m : float = 1, s : int | float = 0.5, beta : float = 1, reduction : str = 'mean') -> torch.Tensor:
    """
    Implements the OTW distance between two time series, as defined in equation (10) of the paper.

    Parameters
    ----------
    x : torch.Tensor
        First time series, of shape (B, L) where B is the batch size and L is the length of the time series.
    y : torch.Tensor
        Second time series, of shape (B, L) where B is the batch size and L is the length of the time series.
    m : float
        Waste cost parameter, default is 1.
    s : int | float
        Window size parameter, it can be an integer or a float between 0 and 1. Default is 0.5.
        If float, it is interpreted as a fraction of the length of the time series.
        If integer, it is interpreted as the number of time steps.
    beta : float
        Hyperparameter for the smooth l1 loss, default is 1.

    Returns
    -------
    torch.Tensor
        OTW distance between the two time series
    """
    
    # Set temporary to high value to stabilize training
    # beta = 100

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Check inputs

    if x.dim() != y.dim() :
        raise ValueError(f"Input time series must have the same number of dimensions. Current dimensions: x {x.shape}, y {y.shape}")

    if x.dim() > 2 or y.dim() > 2 :
        raise ValueError(f"Input time series must be 2-dimensional (B, L). Current dimensions: x {x.shape}, y {y.shape}")
    elif x.dim() == 1 and y.dim() == 1 :
        # Handle the case of single time series (no batch dimension)
        x = x.unsqueeze(0)
        y = y.unsqueeze(0)

    if s <= 0 :
        raise ValueError(f"Window size parameter s must be positive. Current value: {s}")

    if 0 < s < 1 :
        s = int(s * x.size(1))

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Compute OTW distance

    otw_term_1 = m * smooth_l1_loss(window_cumsum(x - y, s), beta, reduction = reduction)

    otw_term_2 = 0
    for i in range(x.shape[1] - 1) : 
        otw_term_2 += smooth_l1_loss(window_cumsum(x[:, 0:(i + 1)] - y[:, 0:(i + 1)], s), beta, reduction = reduction)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    return otw_term_1 + otw_term_2


def smooth_l1_loss(x : torch.Tensor, beta : float, reduction = 'mean') -> torch.Tensor :
    """
    Computes the smooth l1 of the input tensor x, as defined in equation (9) of the paper.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor of shape (B). Each element corresponds to the difference between two time series.
    beta : float
        Hyperparameter for the smooth l1 loss.
    reduction : str
        Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'. Default: 'mean'.

    Returns
    -------
    torch.Tensor
        Smooth l1 loss of the input tensor. If reduction is 'none', the output has the same shape as x. If reduction is 'mean' or 'sum', the output is a scalar.
    """
    
    # Compute smooth l1 loss element-wise
    loss = torch.where(torch.abs(x) < beta, 0.5 * x ** 2 / beta, torch.abs(x) - 0.5 * beta)
    
    # Apply reduction
    if reduction == 'mean' :
        loss = loss.mean()
    elif reduction == 'sum' :
        loss = loss.sum()

    return loss

def window_cumsum(x : torch.Tensor, s : int) -> torch.Tensor :
    """
    Computes the cumulative sum of the input tensor x as defined in equation (7) of the paper.

    Given a time series A represented as an array of values [a1, a2, ..., aL], the window cumsum is computed as :
    window_cumsum(A) = cumsum(A) - cumsum(A[0:L-s])
    (i.e. the cumsum of all the array minus the cumsum of the array excluding the last s elements)

    Parameters
    ----------
    x : torch.Tensor
        Input tensor of shape (B, L).
    s : int
        Window size.

    Returns
    -------
    torch.Tensor
        Cumulative sum over the sliding window, of shape (B).
    """

    cumsum_x = torch.cumsum(x, dim = 1)
    
    windowed_cumsum = cumsum_x[:, -1] - (cumsum_x[:, - (s + 1)] if s < x.shape[1] else 0)

    return windowed_cumsum


