import math
import itertools

import torch

# Define EPS for numerical stability
eps = 7.0/3 - 4.0/3 - 1


def getloss(muG, sigmaG, muM, sigmaM):
    r"""PROPEL Loss function
    Implements the following equations:

    In forward pass:
     .. math::
      L = -\log\underbrace{\left[ \frac{2}{I} \sum_{i=1}^{I} G(P_{gt}, P_{i}) \right]\rule[-12pt]{0pt}{5pt}}_{\mbox{$T1$}} + \log \underbrace{\left[H({P_{gt}}) + \frac{1}{I^2}\sum_{i=1}^{I} H({P_{i}})  + \frac{2}{I^2} \sum_{i < j}^{I} G({P_{i},P_{j}}) \right]\rule[-12pt]{0pt}{5pt}}_{\mbox{$T2$}}

    In backward pass:
     .. math::
      \frac{\partial L}{\partial \mu_{x_{ni}}} = -\frac{1}{T1}\left[ \frac{\partial G(P_{gt}, P_{i})}{\partial \mu_{x_{ni}}} \right] + \frac{1}{T2} \left[ \frac{2}{I^2} \sum_{i < j}^{I} \frac{\partial G({P_{i},P_{j}})}{\partial \mu_{x_{ni}}} \right]
     .. math:: 
      \frac{\partial L}{\partial \sigma_{x_{ni}}} = -\frac{1}{T1}\left[ \frac{\partial G(P_{gt}, P_{i})}{\partial \sigma_{x_{ni}}} \right] + \frac{1}{T2} \left[ \frac{1}{I^2} \frac{\partial H({P_{i}})}{\partial \sigma_{x_{ni}}} + \frac{2}{I^2} \sum_{i < j}^{I} \frac{\partial G({P_{i},P_{j}})}{\partial \sigma_{x_{ni}}} \right]

    Args:
        muG (torch.tensor): mean for groundtruth Gaussian distribution
        sigmaG (torch.tensor): standard deviation for groundtruth Gaussian distribution
        muM (torch.tensor): mean for model Mixture of Gaussian distribution (model output)
        sigmaM (torch.tensor): standard deviation for Mixture of Gaussian distribution (model output)

    Returns:
        torch.tensor: computed loss in forward pass, gradients w.r.t muM/sigmaM in backward pass
    """

    # get the number of gaussians in the mixture model
    num_gaussians = muM.shape[1]

    # \frac{2}{I} \sum_{i=1}^{I} G(P_{gt}, P_{i})
    T1 = ((2 / num_gaussians) * g_function(muG, sigmaG, muM, sigmaM)
          ).sum(dim=-1)  # around num_gaussians

    # H({P_{gt}}) + \frac{1}{I^2}\sum_{i=1}^{I} H({P_{i}})
    T2 = h_function(sigmaG) + ((1/num_gaussians**2)
                               * h_function(sigmaM)).sum(dim=-1)

    # computing the last term of a^2 + b^2 + >>>2ab <<<<<<
    # \frac{2}{I^2} \sum_{i < j}^{I} G({P_{i},P_{j}}) 
    i_index = torch.tensor(
        [i for i, j in itertools.combinations(range(num_gaussians), 2)])
    j_index = torch.tensor(
        [j for i, j in itertools.combinations(range(num_gaussians), 2)])

    i_index = i_index.to(muM.device)
    j_index = j_index.to(muM.device)

    T2_in = g_function(muM.index_select(1, i_index), sigmaM.index_select(
        1, i_index), muM.index_select(1, j_index), sigmaM.index_select(1, j_index)).sum(dim=1)

    T2 = T2 + (2 / (num_gaussians**2)) * T2_in

    L = -torch.log10(T1) + torch.log10(T2)
    return L, T1, T2


def h_function(sigmaM):
    r"""H function implementation

    Implements the following equation:

     .. math::
       H(P_i) = \frac{1}{(2\sqrt{\pi})^n \sqrt{\sigma_{x_{1i}}\cdots\sigma_{x_{ni}}}}

    Args:
        sigmaM (torch.tensor): standard deviation of our input Gaussian distribution

    Returns:
        torch.tensor: result of H(P_m)
    """
    num_dims = sigmaM.shape[-1]
    dTerm = sigmaM.prod(dim=-1)
    dTerm_all = (2*math.sqrt(math.pi))**(num_dims) * torch.sqrt(dTerm)
    out = 1/(dTerm_all.clamp_min(eps))
    return out


def g_function(muM1, sigmaM1, muM2, sigmaM2):
    r"""G Function implementation

    Implements the following equation:

     .. math::
      G(P_i, P_j) = \frac{e^{\big[\frac{2\mu_{x_{1i}}\mu_{x_{1j}} - {\mu_{x_{1i}}}^2 - {\mu_{x_{1j}}}^2}{2(\sigma_{x_{1i}}+\sigma_{x_{1j}})} + \cdots + \frac{2\mu_{x_{ni}}\mu_{x_{nj}} - {\mu_{x_{ni}}}^2 - {\mu_{x_{nj}}}^2}{2(\sigma_{x_{ni}}+\sigma_{x_{nj}})}\big]}}{(\sqrt{2\pi})^n \sqrt{(\sigma_{x_{1i}} + \sigma_{x_{1j}}) \cdots (\sigma_{x_{ni}} + \sigma_{x_{nj}})}}

    Args:
        muM1 (torch.tensor): mean for first Gaussian distribution
        sigmaM1 (torch.tensor): standard deviation for first Gaussian distribution
        muM2 (torch.tensor): mean for second Gaussian distribution
        sigmaM2 (torch.tensor): standard deviation for second Gaussian distribution

    Returns:
        torch.tensor: result of G(P_1, P_2) 
    """
    num_dims = muM1.shape[-1]
    num_gaussians = muM2.shape[1]

    # expand mus and sigmas if not enough recieved
    if len(muM1.shape) < len(muM2.shape):
        muM1 = muM1.unsqueeze(dim=1)

    # calculate the denominator term
    sumSigma = sigmaM1 + sigmaM2
    mulSigma = sumSigma.prod(dim=-1)  # mul dimensions
    aTerm = (math.sqrt(2*math.pi)**(num_dims)) * torch.sqrt(mulSigma)
    A = 1/(aTerm.clamp_min(eps))

    bTerm = 2*(sigmaM1 + sigmaM2)
    B = ((2*muM1*muM2 - muM1.pow(2) - muM2.pow(2)) /
         (bTerm.clamp_min(eps))).sum(dim=-1)

    out = A * torch.exp(B)
    return out


def unpack_prediction(pred, num_dims):
    r"""Helper function to unpack tensor coming from output of neural network

    It expects the pred tensor to have the following shape:
    [num_batch, num_gaussians, num_dimensions * 2] where:

    First [num_batch, num_gaussians, ::num_dimensions] correspond to mean

    Second [num_batch, num_gaussians, num_dimensions::] correspond to standard deviation

    Args:
        pred (torch.tensor): prediction output from a neural network with shape [num_batch, num_gaussians, num_dimensions]
        num_dims (int): number of dimensions to unpack data for, e.g. 3 for 3D problems

    Returns:
        tuple of torch.tensors: unpacked mean (g_mu) and standard deviation (g_sigma)
    """
    # index (num_dimensions) we are splitting is for the last dimension, i.e. 2
    g_mu, g_sigma = torch.split(pred, [num_dims, num_dims], dim=2)

    return g_mu, g_sigma


class PROPEL(torch.nn.modules.loss._Loss):
    r"""PRObabilistic Parametric rEgression Loss (PROPEL) for enabling
    neural networks to output parameters of a mixture of Gaussian distributions from [1].

    [1] "PROPEL: Probabilistic Parametric Regression Loss for Convolutional Neural Networks", 
    M. Asad et al. - 25th International Conference on Pattern Recognition (ICPR), 2020

    Usage instructions:
    In order to use the loss function, the expected output shape from neural network is:
    [num_batches, num_gaussians, 2*num_dimensions] 

    where, 

    num_batches --> number of batches
    num_gaussians --> number of Gaussians in mixture of Gaussians
    num_dimensions --> number of dimensions in each sample - 2 accounts for mean/variance for each dimension

    num_gaussians can be set to a number corresponding to how complex you wish your mixture of Gaussian distribution, e.g. 
    num_gaussians = 2 vs num_gaussians = 10 (where 10 can model much more complex distribution whereas 2 will apply
    regularisation affect by trying to model a gt distribution with 2 Gaussians in mixture)

    One example is of inferring 3D head orientation from 2D images. The output is 3D (num_dimensions = 3) and if we use
    two Gaussians with a num_batch=2, then output will be of size

    [b, 2, 2 * num_dimensions]

    = [b, 2, 6]  - shape for output of the network

    For further usage examples, see optimisation tests inside tests/ folder within this project directory.
    """

    def __init__(self, sigma=0.1, reduction='mean'):
        super(PROPEL, self).__init__(reduction=reduction)

        # defining sigma as parameter so *.to(device) function
        # can be used to change device for it from loss object
        self.sigma = torch.nn.Parameter(
            torch.Tensor([[[sigma]]]), requires_grad=False)

    def forward(self, output, target):

        # target shape:
        # [batch, num_dims]

        # output shape (network output):
        # [batch, num_gaussians, 2 * num_dims]
        #
        #   2 is to account for mean and variance
        # first     [batch, num_gaussians, num_dims] correspond to mean
        # second    [batch, num_gaussians, num_dims] correspond to variance

        # get size of everything
        num_batch = output.shape[0]
        num_dims = int(output.shape[2]/2)

        # check if target labels are in line with what we got
        assert num_dims == target.shape[1], 'Num of dimensions dont match : output [%d] != target [%d]' % (
            num_dims, target.shape[1])

        # split output prediction into relevant sections
        # splits are of size num_dims (mean) and num_dims (variance)
        output_mu, output_sigma = unpack_prediction(output, num_dims)

        # apply differentiable loss - forward + backwards
        L, _, _ = getloss(target, self.sigma.pow(
            2), output_mu, output_sigma.pow(2))

        # apply selected reduction method
        if self.reduction == 'mean':
            return L.mean()
        elif self.reduction == 'sum':
            return L.sum()
        else:  # none
            return L
