import math
import time
import unittest
from functools import wraps

import numpy as np
import torch

from torchpropel import PROPEL, h_function, g_function

# Define EPS for numerical stability
eps = 7.0/3 - 4.0/3 - 1

# set deterministic seed
torch.manual_seed(15)
np.random.seed(15)

# define some decorators for optimisation tests

# helper functions/decorators should not have name test in it - if it is not directly a test:
# https://stackoverflow.com/q/1120148

def skip_if_no_cuda(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if kwargs['device'].type == 'cuda':
            if torch.cuda.is_available():
                return fn(*args, **kwargs)
            else:
                raise unittest.SkipTest('skipping as cuda device not found')
        else:  # cpu
            return fn(*args, **kwargs)
    return wrapper


def exec_time(fn):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = fn(*args, **kwargs)
        if 'num_gaussians' in kwargs.keys() and 'num_dims' in kwargs.keys() and 'device' in kwargs.keys():
            print('Test %s() with num_gaussians=%d, num_dims=%d on %s took %.2f s' % (
                fn.__name__, kwargs['num_gaussians'], kwargs['num_dims'], kwargs['device'], time.time() - start))
        elif 'device' in kwargs.keys():
            print('Test %s() with %s took %.2f s' %
                  (fn.__name__, kwargs['device'], time.time() - start))
        else:
            print('Test %s() took %.2f s' % (fn.__name__, time.time() - start))
        return result

    return wrapper


@skip_if_no_cuda
@exec_time
def propel_loss_layer_optimisation_eval(num_gaussians, num_dims, device=torch.device('cpu')):
    num_batch = 16

    propel_loss = PROPEL(0.2).to(device)

    y = torch.rand((num_batch, num_dims), device=device) + \
        torch.tensor(1., device=device)

    feat_g = np.random.rand(num_batch, num_gaussians, 2 * num_dims)
    feat_g[:, :, num_dims::] = 1.0
    feat = torch.tensor(feat_g, dtype=y.dtype,
                        device=device, requires_grad=True)

    num_epochs = 1000
    for i in range(num_epochs):
        out = propel_loss(feat, y)
        out.backward()

        with torch.no_grad():
            feat -= torch.tensor(0.1, device=device) * feat.grad

        feat.grad.zero_()

    # loss below 0.3 means we are going towards convergence
    return out.detach().cpu().numpy() < 0.3


@skip_if_no_cuda
@exec_time
def propel_compute_loss(output, target, sigma, device=torch.device('cpu')):
    output = output.to(device)
    target = target.to(device)
    propel_loss = PROPEL(sigma).to(device)
    return propel_loss(output, target)


def h_function_numpy(sigmaM):
    num_dims = sigmaM.shape[-1]
    dTerm = sigmaM.prod(axis=-1)
    dTerm_all = (2*math.sqrt(math.pi))**(num_dims) * np.sqrt(dTerm)
    out = 1/(np.clip(dTerm_all, a_min=eps, a_max=None))
    return out


@skip_if_no_cuda
@exec_time
def check_h_function(device=torch.device('cpu')):
    num_batch = 16
    num_dims = 3

    target = torch.rand(num_batch, num_dims, device=device)

    hfout = h_function(target)
    hfout_np = h_function_numpy(target.detach().cpu().numpy())

    return np.allclose(hfout.detach().cpu().numpy(), hfout_np)


def g_function_numpy(muM1, sigmaM1, muM2, sigmaM2):
    num_dims = muM1.shape[-1]
    num_gaussians = muM2.shape[1]

    # calculate the denominator term
    sumSigma = sigmaM1 + sigmaM2
    mulSigma = sumSigma.prod(axis=-1)  # mul dimensions
    aTerm = (math.sqrt(2*math.pi)**(num_dims)) * np.sqrt(mulSigma)
    A = 1/(np.clip(aTerm, a_min=eps, a_max=None))

    bTerm = 2*(sigmaM1 + sigmaM2)
    B = ((2*muM1*muM2 - muM1**2 - muM2**2) /
         (np.clip(bTerm, a_min=eps, a_max=None)).sum(axis=-1))

    out = A * np.exp(B)
    return out


@skip_if_no_cuda
@exec_time
def check_g_function(device=torch.device('cpu')):
    num_batch = 16
    num_dims = 3

    target_mu = torch.rand(num_batch, 1, num_dims, device=device)
    target_sigma = torch.rand(num_batch, 1, num_dims, device=device)

    gfout = g_function(target_mu, target_sigma, target_mu, target_sigma)
    gfout_np = g_function_numpy(target_mu.cpu().numpy(), target_sigma.cpu().numpy(),
                             target_mu.cpu().numpy(), target_sigma.cpu().numpy())

    return np.allclose(gfout.detach().cpu().numpy(), gfout_np)


class PROPELTest(unittest.TestCase):

    def test_propel_loss_optimisation_cpu(self):
        """PROPEL optimisation test with num_gaussians=various, num_dimensions=various on CPU

        Initialises a single layer dummy network with random variables
        and then optimises those to check whether they converge to groundtruth values
        """
        self.assertTrue(propel_loss_layer_optimisation_eval(
            num_gaussians=3, num_dims=3, device=torch.device('cpu')))
        
        self.assertTrue(propel_loss_layer_optimisation_eval(
            num_gaussians=6, num_dims=3, device=torch.device('cpu')))
        
        self.assertTrue(propel_loss_layer_optimisation_eval(
            num_gaussians=3, num_dims=6, device=torch.device('cpu')))

    def test_propel_loss_optimisation_cuda(self):
        """PROPEL optimisation test with num_gaussians=various, num_dimensions=various on GPU

        Initialises a single layer dummy network with random variables
        and then optimises those to check whether they converge to groundtruth values
        """
        self.assertTrue(propel_loss_layer_optimisation_eval(
            num_gaussians=3, num_dims=3, device=torch.device('cuda')))
        
        self.assertTrue(propel_loss_layer_optimisation_eval(
            num_gaussians=6, num_dims=3, device=torch.device('cuda')))
        
        self.assertTrue(propel_loss_layer_optimisation_eval(
            num_gaussians=3, num_dims=6, device=torch.device('cuda')))

    def test_propel_loss_zero_cpu(self):
        """PROPEL sanity test to check if loss equals zero for gt=pred 
        Runs on CPU
        """
        device = torch.device('cpu')

        sigma = 0.2

        num_batch = 16
        num_dims = 3
        num_gaussians = 6

        target = torch.ones((num_batch, num_dims)) * 0.5

        output_g = np.ones((num_batch, num_gaussians, 2 * num_dims)) * 0.5
        output_g[:, :, num_dims::] = 0.2
        output = torch.tensor(output_g, dtype=target.dtype)

        # propel_loss = PROPEL(0.2)
        # L = propel_loss(output, target)
        L = propel_compute_loss(output, target, sigma, device=device)

        self.assertTrue(L < 0.000001)

    def test_propel_loss_zero_cuda(self):
        """PROPEL sanity test to check if loss equals zero for gt=pred 
        Runs on GPU
        """
        device = torch.device('cpu')

        sigma = 0.2

        num_batch = 16
        num_dims = 3
        num_gaussians = 6

        target = torch.ones((num_batch, num_dims)) * 0.5

        output_g = np.ones((num_batch, num_gaussians, 2 * num_dims)) * 0.5
        output_g[:, :, num_dims::] = 0.2
        output = torch.tensor(output_g, dtype=target.dtype)

        # propel_loss = PROPEL(0.2)
        # L = propel_loss(output, target)
        L = propel_compute_loss(output, target, sigma, device=device)

        self.assertTrue(L < 0.000001)

    def test_propel_loss_large_cpu(self):
        """PROPEL sanity test to check if loss equals inf for gt!=pred 
        Runs on CPU
        """
        device = torch.device('cpu')

        sigma = 0.2

        num_batch = 16
        num_dims = 3
        num_gaussians = 6

        target = torch.ones((num_batch, num_dims)) * 0.5 + 10

        output_g = np.ones((num_batch, num_gaussians, 2 * num_dims)) * 0.5
        output_g[:, :, num_dims::] = 0.2
        output = torch.tensor(output_g, dtype=target.dtype)

        # propel_loss = PROPEL(0.2)
        # L = propel_loss(output, target)
        L = propel_compute_loss(output, target, sigma, device=device)

        self.assertTrue(torch.isinf(L))


    def test_propel_loss_large_cuda(self):
        """PROPEL sanity test to check if loss equals inf for gt!=pred 
        Runs on GPU
        """
        device = torch.device('cpu')

        sigma = 0.2

        num_batch = 16
        num_dims = 3
        num_gaussians = 6

        target = torch.ones((num_batch, num_dims)) * 0.5 + 10

        output_g = np.ones((num_batch, num_gaussians, 2 * num_dims)) * 0.5
        output_g[:, :, num_dims::] = 0.2
        output = torch.tensor(output_g, dtype=target.dtype)

        # propel_loss = PROPEL(0.2)
        # L = propel_loss(output, target)
        L = propel_compute_loss(output, target, sigma, device=device)

        self.assertTrue(torch.isinf(L))

    def test_hfunction_cpu(self):
        """PROPEL sanity test to check if hfunction is working as expected
        Runs on CPU
        """
        device = torch.device('cpu')
        self.assertTrue(check_h_function(device=device))

    def test_hfunction_cuda(self):
        """PROPEL sanity test to check if hfunction is working as expected
        Runs on GPU
        """
        device = torch.device('cuda')
        self.assertTrue(check_h_function(device=device))

    def test_gfunction_cpu(self):
        """PROPEL sanity test to check if gfunction is working as expected
        Runs on CPU
        """
        device = torch.device('cpu')
        self.assertTrue(check_g_function(device=device))

    def test_gfunction_cuda(self):
        """PROPEL sanity test to check if gfunction is working as expected
        Runs on GPU
        """
        device = torch.device('cuda')
        self.assertTrue(check_g_function(device=device))


if __name__ == '__main__':
    unittest.main()
