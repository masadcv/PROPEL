import torch
import numpy as np

from torchpropel import PROPEL

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
