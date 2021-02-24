# PRObablistic Parametric rEgression Loss (PROPEL) 
PRObabilistic Parametric rEgresison Loss (PROPEL) is a loss function that enables probabilisitic regression for a neural network. It achieves this by enabling a neural network to learn parameters of a mixture of Gaussian distribution. 

Further details about the loss can be found in the paper: [PROPEL: Probabilistic Parametric Regression Loss for Convolutional Neural Networks](https://arxiv.org/pdf/1807.10937.pdf)

This repository provides official pytorch implementation of PROPEL. 

# Installation Instructions
PROPEL can be installed using the following command 
```bash
pip install git+https://github.com/masaddev/PROPEL.git
```

# Usage Example
```python
import torch
import numpy as np

from propel_loss import PROPEL

# Our example has a neural network with
# output [num_batch, num_gaussians, num_dims]
num_batch = 4
num_gaussians = 6
num_dims = 3

# setting ground-truth variance sigma_gt=0.2
sigma_gt = 0.2
propel_loss = PROPEL(sigma_gt)

# ground truth targets for loss
y = torch.ones((num_batch, num_dims)) * 0.5

# example prediction - this can also be coming as output of a neural network
feat_g = np.random.randn(num_batch, num_gaussians, 2 * num_dims) * 0.5
feat_g[:, :, num_dims::] = 0.2
feat = torch.tensor(feat_g, dtype=y.dtype)

# compute the loss
L = propel_loss(feat, y)

print(L)
```
# Documentation
Further details of each function implemented for PROPEL can be accessed at the documentation hosted at: [https://masaddev.github.io/PROPEL/index.html](https://masaddev.github.io/PROPEL/index.html). 

# Citing PROPEL
Pre-print of PROPEL can be found at: [PROPEL: Probabilistic Parametric Regression Loss for Convolutional Neural Networks](https://arxiv.org/pdf/1807.10937.pdf)

If you use PROPEL in your research, then please cite:

BibTeX:
```
@inproceedings{asad2020propel,
  title={PROPEL: Probabilistic Parametric Regression Loss for Convolutional Neural Networks},
  author={Asad, Muhammad and Basaru, Rilwan and Arif, SM and Slabaugh, Greg},
  booktitle={25th International Conference on Pattern Recognition},
  pages={},
  year={2020}}
```
