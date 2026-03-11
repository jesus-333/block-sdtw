"""
dtw_loss_functions
==================

A package for computing DTW-based loss functions for PyTorch models.

Modules
-------
block_dtw
    Block-wise variant of the Soft DTW loss.
otw
    Optimal Transport Warping loss.
soft_dtw
    Soft Dynamic Time Warping loss. This module is a wrapper for the various implementations of the SoftDTW available online, which are collected in the ``soft_dtw_implementations`` subpackage.

Subpackages
-----------
soft_dtw_implementations
    Collection of various implementations of the SoftDTW loss available online.
"""

__author__    = "Alberto (Jesus) Zancanaro"
__email__     = "alberto.zancanaro@uni.lu"
__version__   = "1.0.4"
__license__   = "Apache-2.0"
