# DTW Inspired Loss function

This repository contains the implementation of various DTW inspired loss function.
Each implementation is compatible with PyTorch and can be used for training.

## Currently Implemented
- SoftDTW. CUDA implementation of the SoftDTW, by Mehran Maghoumi. See the [original repository](https://github.com/Maghoumi/pytorch-softdtw-cuda) for more details.
- BlockDTW. Alternative version of SDTW, where a block-wise computation is used to improve performance. See the paper "BlockDTW: Efficient and Scalable Similarity Search Algorithm for Healthcare-Focused Time-Series" for more details.
- OTW

