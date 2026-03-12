# DTW Inspired Loss Functions

Python package with the implementation of various DTW-inspired loss functions.
Each implementation is compatible with PyTorch and can be used for training.

### Loss Functions Currently Implemented
- **SDTW**. CUDA implementation of the SofDTW algorithm. This package offers several implementations of this algorithm. Currently
  - [pytorch-softdtw-cuda](https://github.com/Maghoumi/pytorch-softdtw-cuda) by Mehran Maghoumi
  - [pysdtw](https://github.com/toinsson/pysdtw) by Antoine Loriette
  - [sdtw-cuda-torch](https://github.com/BGU-CS-VIL/sdtw-cuda-torch) by BGU-CS-VIL (implemented by Ron Shapira Weber)
- **BlockDTW**. Alternative version of SDTW, where a block-wise computation is used to improve performance. See the paper [BlockDTW: Efficient and Scalable Similarity Search Algorithm for Healthcare-Focused Time-Series](https://ieeexplore.ieee.org/document/11230700) for more details.
- **OTW**. Implementation of the Optimal Transport Warping function. See the paper [OTW: Optimal Transport Warping for Time Series](https://ieeexplore.ieee.org/document/10095915) for more details.


### Documentation

You can read the package documentation at the [link](https://jesus-333.github.io/dtw_loss_functions_documentation/)

Note that for now the documentation is generated automatically from the docstrings in the code, through sphinx autodoc. 
It will be improved in the future, with more examples and explanations.


## Installation 

### pip installation
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

## Usage Examples
Each loss function is implemented as a module inside the package.

### Block-DTW Example
```python
from dtw_loss_functions import block_dtw
import torch

block_size = 25
use_cuda = torch.cuda.is_available()
block_dtw_loss = block_dtw.block_dtw(block_size, sdtw_config = {'use_cuda' : use_cuda})

batch_size = 5
time_samples = 300
channels = 1

device = 'cuda' if use_cuda else 'cpu'
x   = torch.randn(batch_size, time_samples, channels).to(device)
x_r = torch.randn(batch_size, time_samples, channels).to(device)

output_block_dtw = block_dtw_loss(x, x_r)
```

### SoftDTW Example

This package offer several implementations for the SoftDTW algorithm.

Mehran Maghoumi's (`mag`) implementation
```python
import torch
from dtw_loss_functions import soft_dtw

use_cuda = torch.cuda.is_available()

sdtw_loss = soft_dtw.soft_dtw(implementation = 'mag', sdtw_config = {'use_cuda' : use_cuda, 'gamma' : 0.1})

batch_size = 5
time_samples = 300
channels = 1

device = 'cuda' if use_cuda else 'cpu'
x   = torch.randn(batch_size, time_samples, channels).to(device)
x_r = torch.randn(batch_size, time_samples, channels).to(device)

output_sdtw = sdtw_loss(x, x_r)
```

Ron Shapira Weber's (`ron`) implementation
```python
import torch
from dtw_loss_functions import soft_dtw

sdtw_loss = soft_dtw.soft_dtw(implementation = 'ron', sdtw_config = {'gamma' : 0.1, 'dist' : 'sqeuclidean'})

batch_size = 5
time_samples = 300
channels = 1

device = 'cuda' if torch.cuda.is_available() else 'cpu'
x   = torch.randn(batch_size, time_samples, channels).to(device)
x_r = torch.randn(batch_size, time_samples, channels).to(device)

output_sdtw = sdtw_loss(x, x_r)
```

# Citation 

If you use this package refer to the [citation file](https://github.com/jesus-333/dtw_loss_functions/blob/main/citations.md) for all the info regarding the works to cite.

# Important notes about Repository Name
The original name of this repository was `block-sdtw` because it was initially created to develop only the code related to block-DTW loss function. As with many academic projects, the code was initially "not very well organized", so I decided to refactor it and transform it into a Python package for easier use.
Since my implementation of Block-DTW relies on Mehran Maghoumi's [implementation](https://github.com/Maghoumi/pytorch-softdtw-cuda) of SoftDTW, I decided to also include SoftDTW inside the package.
Due to some testing, I also had the opportunity to implement the [OTW](https://ieeexplore.ieee.org/document/10095915) loss function.

At this point, given the presence of 3 different loss functions within the code base, the name `block-sdtw` for the package seemed a bit misleading to me. So, I decided to change the name from `block-sdtw` to `dtw-loss-function`.
