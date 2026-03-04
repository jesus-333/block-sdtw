"""
Script to check the correct functioning of optimized block SDTW implementation and compare it with the naive implementation.
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

import numpy as np
import torch

import block_sdtw
import soft_dtw_cuda

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

batch_size = 5
time_samples = 100
channels = 1

block_size = 25

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

if torch.cuda.is_available() :
    device = torch.device("cuda")
    print("CUDA backend in use\n")
else:
    device = torch.device("cpu")
    print("No backend in use. Device set to cpu\n")

# Initialize SDTW loss function
recon_loss_function = soft_dtw_cuda.SoftDTW(use_cuda = True if device.type == 'cuda' else False)

# Generate synthetic data
x   = torch.randn(batch_size, time_samples, channels).to(device)
x_r = torch.randn(batch_size, time_samples, channels).to(device)

print("Note that the data will be always the same for all the experiments\n")

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Test 1 : soft dtw channels

# Compute loss
output = recon_loss_function(x, x_r)

# Check results
print("First experiment (soft dtw)")
print(f"x shape     : {x.shape}")
print(f"x_r shape   : {x_r.shape}")
print(f"Output shape: {output.shape}")
print(f"Output      : {output}")
print(f"Output mean : {output.mean()}")

# Reshape data to have multiple channels
if time_samples % block_size != 0 : raise ValueError(f"time_samples must be multiple of block_size. Current values are time_samples = {time_samples}, block_size = {block_size}. Rest of the division is {time_samples % block_size}")
channels = int(time_samples / block_size)

x_reshaped   = x.view(batch_size, block_size, channels)
x_r_reshaped = x_r.view(batch_size, block_size, channels)

# Compute loss
output_reshaped = recon_loss_function(x_reshaped, x_r_reshaped)

# Check results
print("\nSecond experiment (soft dtw) (reshaped data)")
print(f"x_reshaped shape     : {x_reshaped.shape}")
print(f"x_r_reshaped shape   : {x_r_reshaped.shape}")
print(f"Output_reshaped shape: {output_reshaped.shape}")
print(f"Output_reshaped      : {output_reshaped}")
print(f"Output_reshaped mean : {output_reshaped.mean()}")

print("\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n")

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Test 2 : block SDTW optimization

# Compute loss
output_block_sdtw = block_sdtw.block_sdtw(x, x_r, recon_loss_function, block_size, soft_DTW_type = 3)
output_block_sdtw_optimized = block_sdtw.block_sdtw_optimized(x, x_r, recon_loss_function, block_size, soft_DTW_type = 3)

# Check results
print("\nThird experiment (block sdtw)")
print(f"x shape     : {x.shape}")
print(f"x_r shape   : {x_r.shape}")
print(f"Output_block_sdtw shape          : {output_block_sdtw.shape}")
print(f"Output_block_sdtw_optimized shape: {output_block_sdtw_optimized.shape}")
print(f"Output_block_sdtw                : {output_block_sdtw}")
print(f"Output_block_sdtw_optimized      : {output_block_sdtw_optimized}")
print(f"Output_block_sdtw mean           : {output_block_sdtw.mean()}")
print(f"Output_block_sdtw_optimized mean : {output_block_sdtw_optimized.mean()}")

