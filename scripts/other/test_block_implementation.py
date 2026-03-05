"""
Script to check the correct functioning of optimized block DTW implementation and compare it with the naive implementation.
The script will generate synthetic data and compute the loss with both implementations, checking the results and comparing them (i.e. checking if shapes and numerical values are consistent between the two implementations).
Furthermore, the script will also evaluate the computational time of the two block DTW implementations and of the SDTW loss function and compare them.

You can launch this script with the python command or run it from and IDE. 

If you launch it from the command line, you can specify the following arguments:
- --batch_size: Batch size for the input data (default: 5)
- --time_samples: Number of time samples for the input data (default: 300)
- --channels: Number of channels for the input data (default: 1)
- --block_size: Block size for the block DTW implementation (default: 25)
- --use_cuda: Whether to use CUDA backend for the computations. If not specified, it will check automatically if CUDA is available and use it if possible (default: None)
If an argument is not specified, the default value will be used (the default values are specified inside the script).
Note that when defined, arguments specified in the command line will always override the default values specified in the script.

If you launch the script from an IDE, you can specify the arguments in the "Default arguments" section of the script, by changing the values of the variables "default_batch_size", "default_time_samples", "default_channels", "default_block_size" and "default_use_cuda". 

Note that due to constraints of the block DTW implementation, the number of time samples must be a multiple of the block size. If this is not the case, the script will raise an error.

@author: Alberto Zancanaro (Jesus)
@organization: Luxembourg Centre for Systems Biomedicine (LCSB)
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Script arguments

import argparse

parser = argparse.ArgumentParser(description = 'Test block DTW implementation')

parser.add_argument('--batch_size'     , type = int, default = None, help = 'Batch size for the input data')
parser.add_argument('--time_samples'   , type = int, default = None, help = 'Number of time samples for the input data')
parser.add_argument('--channels'       , type = int, default = None, help = 'Number of channels for the input data')
parser.add_argument('--block_size'     , type = int, default = None, help = 'Block size for the block DTW implementation')
parser.add_argument('--n_repetitions'  , type = int, default = None, help = 'Number of repetitions for the time evaluation of the loss functions')
parser.add_argument('--use_cuda'       , action = 'store_true', default = None, help = 'Whether to use CUDA backend for the computations. If None, it will check automatically if CUDA is available and use it if possible')
parser.add_argument('--time_in_seconds', action = 'store_true', default = False, help = 'If set, the computational times will be printed in seconds instead of milliseconds')

args = parser.parse_args()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

import numpy as np
import torch
import time

from dtw_loss_functions import soft_dtw_cuda, block_dtw

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Default arguments

default_batch_size = 5
default_time_samples = 300
default_channels = 1
default_block_size = 25
default_n_repetitions = 100
default_use_cuda = None
default_time_in_seconds = False

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Parameters and device setup

# Set up parameters
batch_size      = args.batch_size if args.batch_size is not None else default_batch_size
time_samples    = args.time_samples if args.time_samples is not None else default_time_samples
channels        = args.channels if args.channels is not None else default_channels
block_size      = args.block_size if args.block_size is not None else default_block_size
n_repetitions   = args.n_repetitions if args.n_repetitions is not None else default_n_repetitions
use_cuda        = args.use_cuda if args.use_cuda is not None else default_use_cuda
time_in_seconds = args.time_in_seconds if args.time_in_seconds is not None else default_time_in_seconds

if time_samples % block_size != 0 :
    raise ValueError(f"time_samples must be multiple of block_size. Current values are time_samples = {time_samples}, block_size = {block_size}. Rest of the division is {time_samples % block_size}")

# Set up device (CUDA or CPU)
if use_cuda is None :
    use_cuda = torch.cuda.is_available()
else :
    if use_cuda and not torch.cuda.is_available() :
        print("Warning: CUDA backend requested but not available. Using CPU instead.\n")
        use_cuda = False
device = torch.device('cuda' if use_cuda else 'cpu')

# Set up loss functions
sdtw_loss = soft_dtw_cuda.SoftDTW(use_cuda = use_cuda)
block_naive_loss = block_dtw.block_dtw_naive(block_size, use_cuda)
block_optimized_loss = block_dtw.block_dtw_optimized(block_size, use_cuda)

# Generate synthetic data
x   = torch.randn(batch_size, time_samples, channels).to(device)
x_r = torch.randn(batch_size, time_samples, channels).to(device)

print(f"Using block size {block_size} for the block DTW implementation")
print(f"Device used for computations: {'CUDA' if use_cuda else 'CPU'}")
print(f"Generated synthetic (x and x_r) data with shape {batch_size} x {time_samples} x {channels}")
print("Note that the same tensors will be always the same for all the experiments")
print("\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n")

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Test 1 : Check if the two block DTW implementations give the same results

# Compute loss
output_block_naive     = block_naive_loss(x, x_r)
output_block_optimized = block_optimized_loss(x, x_r)

# Check results
print("Test 1 : Check if the two block DTW implementations give the same results")
print(f"x shape   : {x.shape}")
print(f"x_r shape : {x_r.shape}")
print("Output SHAPE :")
print(f"\tNaive Impementation         : {output_block_naive.shape}")
print(f"\tOptimized Impementation     : {output_block_optimized.shape}")
print("Output VALUES :")
print(f"\tNaive Implementation          : {output_block_naive}")
print(f"\tNaive Implementation MEAN     : {output_block_naive.mean()}")
print(f"\tOptimized Implementation      : {output_block_optimized}")
print(f"\tOptimized Implementation MEAN : {output_block_optimized.mean()}")

print("\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n")

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Test 2 : Compare the computational time of the two block DTW implementations and of the SDTW loss function

# List to store the computational times for each loss function
block_naive_times     = []
block_optimized_times = []
sdtw_times            = []

# Time computations
for i in range(n_repetitions) :
    start_time = time.time()
    block_naive_loss(x, x_r)
    block_naive_times.append(time.time() - start_time)

    start_time = time.time()
    block_optimized_loss(x, x_r)
    block_optimized_times.append(time.time() - start_time)

    start_time = time.time()
    sdtw_loss(x, x_r)
    sdtw_times.append(time.time() - start_time)

# Convert times to milliseconds
if not time_in_seconds :
    block_naive_times     = np.array(block_naive_times) * 1000
    block_optimized_times = np.array(block_optimized_times) * 1000
    sdtw_times            = np.array(sdtw_times) * 1000

label_time = 's' if time_in_seconds else 'ms'

# Print results
print("Test 2 : Compare the computational time")
print(f"Average computational time over {n_repetitions} repetitions :")
print(f"\tBlock DTW Naive Implementation     : {block_naive_times.mean():.4f}±{block_naive_times.std():.4f} {label_time}")
print(f"\tBlock DTW Optimized Implementation : {block_optimized_times.mean():.4f}±{block_optimized_times.std():.4f} {label_time}")
print(f"\tSDTW Loss Function                 : {sdtw_times.mean():.4f}±{sdtw_times.std():.4f} {label_time}")
