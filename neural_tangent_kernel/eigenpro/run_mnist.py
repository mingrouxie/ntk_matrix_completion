'''Use case: MNIST'''

import mnist
import torch

import eigenpro
import pdb




def euclidean_distances(samples, centers, squared=True):
    """Calculate the pointwise distance.
    Args:
        samples: of shape (n_sample, n_feature).
        centers: of shape (n_center, n_feature).
        squared: boolean.
    Returns:
        pointwise distances (n_sample, n_center).
    """
    samples_norm = torch.sum(samples ** 2, dim=1, keepdim=True)
    if samples is centers:
        centers_norm = samples_norm
    else:
        centers_norm = torch.sum(centers ** 2, dim=1, keepdim=True)
    centers_norm = torch.reshape(centers_norm, (1, -1))

    distances = samples.mm(torch.t(centers))
    distances.mul_(-2)
    distances.add_(samples_norm)
    distances.add_(centers_norm)
    if not squared:
        distances.clamp_(min=0)
        distances.sqrt_()

    return distances


def gaussian(samples, centers, bandwidth):
    """Gaussian kernel.
    Args:
        samples: of shape (n_sample, n_feature).
        centers: of shape (n_center, n_feature).
        bandwidth: kernel bandwidth.
    Returns:
        kernel matrix of shape (n_sample, n_center).
    """
    assert bandwidth > 0
    kernel_mat = euclidean_distances(samples, centers)
    kernel_mat.clamp_(min=0)
    gamma = 1.0 / (2 * bandwidth ** 2)
    kernel_mat.mul_(-gamma)
    kernel_mat.exp_()
    return kernel_mat




n_class = 10
(x_train, y_train), (x_test, y_test) = mnist.load()
x_train, y_train, x_test, y_test = x_train.astype('float32'), \
    y_train.astype('float32'), x_test.astype('float32'), y_test.astype('float32')
pdb.set_trace()

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

kernel_fn = lambda x, y: gaussian(x, y, bandwidth=5)
model = eigenpro.FKR_EigenPro(kernel_fn, x_train, n_class, device=device)
_ = model.fit(x_train, y_train, x_test, y_test, epochs=[1, 2, 5], mem_gb=12)
