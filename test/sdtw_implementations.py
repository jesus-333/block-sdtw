"""
Test the different implementations of the SoftDTW loss function.
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

import time
import torch

from dtw_loss_functions import soft_dtw

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# List of implementations to test
implementations_list = ['mag', 'pysdtw', 'ron']

# Shape of the input tensors
B = 5
T = 300
C = 1

# Repetitions for timing
n_repetitions = 100

time_in_milliseconds = True

sdtw_config = dict(
    gamma = 1,
    normalize = False,
    bandwidth = None,
    dist_func = None,
    dist = 'sqeuclidean',
    use_cuda = torch.cuda.is_available()
)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# Generate synthetic data
x   = torch.randn(B, T, C)
x_r = torch.randn(B, T, C)

time_dict = {implementation : [] for implementation in implementations_list}

# Test each implementation
for implementation in implementations_list :

    print(f"Testing implementation '{implementation}'")

    # Create the loss function
    sdtw_loss = soft_dtw.soft_dtw(implementation = implementation, sdtw_config = sdtw_config)

    # Compute the loss and time it
    for _ in range(n_repetitions) :

        start_time = time.time()
        output = sdtw_loss(x, x_r)
        end_time = time.time()

        total_time = end_time - start_time
        total_time = total_time * 1000 if time_in_milliseconds else total_time

        time_dict[implementation].append(total_time)

    loss = sdtw_loss(x, x_r)
    end_time = time.time()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Print results

symbol = "ms" if time_in_milliseconds else "s"

print("\nResults:")
print(f"Times are in {symbol} (lower is better)")
print(f"Number of repetitions for timing: {n_repetitions}")
print(f"Input tensors have shape {B} x {T} x {C} (batch size x time samples x channels)")
for implementation in implementations_list :
    times = time_dict[implementation]
    print(f"Implementation '{implementation}':")
    print(f"\tMean time: {torch.tensor(times).mean():.4f} {symbol}")
    print(f"\tStd time : {torch.tensor(times).std():.4f} {symbol}")
    
