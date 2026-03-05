# DTW Inspired Loss Functions

Python package with the implementation of various DTW-inspired loss functions.
Each implementation is compatible with PyTorch and can be used for training.

### Loss Functions Currently Implemented
- SoftDTW. CUDA implementation of the SoftDTW, by Mehran Maghoumi. See the [original repository](https://github.com/Maghoumi/pytorch-softdtw-cuda) for more details.
- BlockDTW. Alternative version of SDTW, where a block-wise computation is used to improve performance. See the paper [BlockDTW: Efficient and Scalable Similarity Search Algorithm for Healthcare-Focused Time-Series](https://ieeexplore.ieee.org/document/11230700) for more details.
- OTW. Implementation of the Optimal Transport Warping function. See the paper [OTW: Optimal Transport Warping for Time Series](https://ieeexplore.ieee.org/document/10095915) for more details.

## Important Notes about Repository Name
This repository was initially created to develop only the code related to block-DTW loss function. As with many academic projects, the code was initially "not very well organized", so I decided to refactor it and transform it into a Python package for easier use.
Since my implementation of Block-DTW relies on Mehran Maghoumi's [implementation](https://github.com/Maghoumi/pytorch-softdtw-cuda) of SoftDTW, I decided to also include SoftDTW inside the package.
Due to some testing, I also had the opportunity to implement the [OTW](https://ieeexplore.ieee.org/document/10095915) loss function.

At this point, given the presence of 3 different loss functions within the code base, the name `block-dtw` for the package seemed a bit misleading to me. So, I decided to change the name from `block-dtw` to `dtw-loss-function`.

## Installation 

### Packet manager
The easiest way to use this package is to install it via pip
```sh
pip install dtw-loss-function
```

### Build from source
Alternatively, you can download the repository and compile it locally via [hatchling](https://pypi.org/project/hatchling/)
```sh
pip install hatchling
git clone https://github.com/jesus-333/dtw_loss_functions.git
cd dtw_loss_functions
hatchling build && pip install .
```

## Usage
Each loss function is implemented as a class inside the packet.

### Block-DTW Example
```python
import torch
from dtw_loss_functions import block_dtw

block_size = 25

use_cuda = torch.cuda.is_available()
device = 'cuda' if use_cuda else 'cpu'

batch_size = 5
time_samples = 300
channels = 1

x   = torch.randn(batch_size, time_samples, channels).to(device)
x_r = torch.randn(batch_size, time_samples, channels).to(device)

block_dtw_loss = block_dtw.block_dtw(block_size, use_cuda)

output_block_dtw = block_dtw_loss(x, x_r)
```

### SoftDTW Example
Note that the SoftDTW CUDA was [originally implemented](https://github.com/Maghoumi/pytorch-softdtw-cuda) by Mehran Maghoumi.
```python
import torch
from dtw_loss_functions import soft_dtw_cuda

use_cuda = torch.cuda.is_available()
device = 'cuda' if use_cuda else 'cpu'

batch_size = 5
time_samples = 300
channels = 1

x   = torch.randn(batch_size, time_samples, channels).to(device)
x_r = torch.randn(batch_size, time_samples, channels).to(device)

sdtw_loss = soft_dtw_cuda.SoftDTW(use_cuda = use_cuda)

output_sdtw = sdtw_loss(x, x_r)
```

