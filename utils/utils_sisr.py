# -*- coding: utf-8 -*-
import numpy as np

from utils import utils_image as util
from utils import utils_deblur

from scipy import ndimage
import scipy
import scipy.stats as ss
import scipy.io as io

from scipy.ndimage import filters, measurements, interpolation
from scipy.interpolate import interp2d


"""
# --------------------------------------------
# Super-Resolution
# --------------------------------------------
#
# Kai Zhang (cskaizhang@gmail.com)
# https://github.com/cszn
# 28/Nov/2019
# --------------------------------------------
"""


"""
# --------------------------------------------
# anisotropic Gaussian kernels
# --------------------------------------------
"""


def anisotropic_Gaussian(ksize=15, theta=np.pi, l1=6, l2=6):
    """ generate an anisotropic Gaussian kernel
    Args:
        ksize : e.g., 15, kernel size
        theta : [0,  pi], rotation angle range
        l1    : [0.1,50], scaling of eigenvalues
        l2    : [0.1,l1], scaling of eigenvalues
        If l1 = l2, will get an isotropic Gaussian kernel.

    Returns:
        k     : kernel
    """

    v = np.dot(np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]), np.array([1., 0.]))
    V = np.array([[v[0], v[1]], [v[1], -v[0]]])
    D = np.array([[l1, 0], [0, l2]])
    Sigma = np.dot(np.dot(V, D), np.linalg.inv(V))
    k = gm_blur_kernel(mean=[0, 0], cov=Sigma, size=ksize)

    return k


def gm_blur_kernel(mean, cov, size=15):
    center = size / 2.0 + 0.5
    k = np.zeros([size, size])
    for y in range(size):
        for x in range(size):
            cy = y - center + 1
            cx = x - center + 1
            k[y, x] = ss.multivariate_normal.pdf([cx, cy], mean=mean, cov=cov)

    k = k / np.sum(k)
    return k


"""
# --------------------------------------------
# calculate PCA projection matrix
# --------------------------------------------
"""


def get_pca_matrix(x, dim_pca=15):
    """
    Args:
        x: 225x10000 matrix
        dim_pca: 15

    Returns:
        pca_matrix: 15x225
    """
    C = np.dot(x, x.T)
    w, v = scipy.linalg.eigh(C)
    pca_matrix = v[:, -dim_pca:].T

    return pca_matrix


def show_pca(x):
    """
    x: PCA projection matrix, e.g., 15x225
    """
    for i in range(x.shape[0]):
        xc = np.reshape(x[i, :], (int(np.sqrt(x.shape[1])), -1), order="F")
        util.surf(xc)


def cal_pca_matrix(path='PCA_matrix.mat', ksize=15, l_max=12.0, dim_pca=15, num_samples=500):
    kernels = np.zeros([ksize*ksize, num_samples], dtype=np.float32)
    for i in range(num_samples):

        theta = np.pi*np.random.rand(1)
        l1    = 0.1+l_max*np.random.rand(1)
        l2    = 0.1+(l1-0.1)*np.random.rand(1)

        k = anisotropic_Gaussian(ksize=ksize, theta=theta[0], l1=l1[0], l2=l2[0])

        # util.imshow(k)

        kernels[:, i] = np.reshape(k, (-1), order="F")  # k.flatten(order='F')

    # io.savemat('k.mat', {'k': kernels})

    pca_matrix = get_pca_matrix(kernels, dim_pca=dim_pca)

    io.savemat(path, {'p': pca_matrix})

    return pca_matrix


"""
# --------------------------------------------
# kernel shift
# --------------------------------------------
"""


def gen_kernel(k_size=np.array([15, 15]), scale_factor=np.array([4, 4]), min_var=0.6, max_var=10., noise_level=0):
    """"
    # modified version of https://github.com/assafshocher/BlindSR_dataset_generator
    # Kai Zhang
    # min_var = 0.175 * sf  # variance of the gaussian kernel will be sampled between min_var and max_var
    # max_var = 2.5 * sf
    """
    # Set random eigen-vals (lambdas) and angle (theta) for COV matrix
    lambda_1 = min_var + np.random.rand() * (max_var - min_var)
    lambda_2 = min_var + np.random.rand() * (max_var - min_var)
    theta = np.random.rand() * np.pi  # random theta
    noise = -noise_level + np.random.rand(*k_size) * noise_level * 2

    # Set COV matrix using Lambdas and Theta
    LAMBDA = np.diag([lambda_1, lambda_2])
    Q = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    SIGMA = Q @ LAMBDA @ Q.T
    INV_SIGMA = np.linalg.inv(SIGMA)[None, None, :, :]

    # Set expectation position (shifting kernel for aligned image)
    MU = k_size // 2 - 0.5*(scale_factor - 1) # - 0.5 * (scale_factor - k_size % 2)
    MU = MU[None, None, :, None]

    # Create meshgrid for Gaussian
    [X,Y] = np.meshgrid(range(k_size[0]), range(k_size[1]))
    Z = np.stack([X, Y], 2)[:, :, :, None]

    # Calcualte Gaussian for every pixel of the kernel
    ZZ = Z-MU
    ZZ_t = ZZ.transpose(0,1,3,2)
    raw_kernel = np.exp(-0.5 * np.squeeze(ZZ_t @ INV_SIGMA @ ZZ)) * (1 + noise)

    # shift the kernel so it will be centered
    #raw_kernel_centered = kernel_shift(raw_kernel, scale_factor)

    # Normalize the kernel and return
    #kernel = raw_kernel_centered / np.sum(raw_kernel_centered)
    kernel = raw_kernel / np.sum(raw_kernel)
    return kernel


"""
# --------------------------------------------
# degradation models
# --------------------------------------------
"""


def bicubic_degradation(x, sf=3):
    '''
    Args:
        x: HxWxC image, [0, 1]
        sf: down-scale factor

    Return:
        bicubicly downsampled LR image
    '''
    x = util.imresize_np(x, scale=1/sf)
    return x


def srmd_degradation(x, k, sf=3):
    ''' blur + bicubic downsampling

    Args:
        x: HxWxC image, [0, 1]
        k: hxw, double
        sf: down-scale factor

    Return:
        downsampled LR image

    Reference:
        @inproceedings{zhang2018learning,
          title={Learning a single convolutional super-resolution network for multiple degradations},
          author={Zhang, Kai and Zuo, Wangmeng and Zhang, Lei},
          booktitle={IEEE Conference on Computer Vision and Pattern Recognition},
          pages={3262--3271},
          year={2018}
        }
    '''
    x = ndimage.filters.convolve(x, np.expand_dims(k, axis=2), mode='wrap')  # 'nearest' | 'mirror'
    x = bicubic_degradation(x, sf=sf)
    return x


def dpsr_degradation(x, k, sf=3):

    ''' bicubic downsampling + blur

    Args:
        x: HxWxC image, [0, 1]
        k: hxw, double
        sf: down-scale factor

    Return:
        downsampled LR image

    Reference:
        @inproceedings{zhang2019deep,
          title={Deep Plug-and-Play Super-Resolution for Arbitrary Blur Kernels},
          author={Zhang, Kai and Zuo, Wangmeng and Zhang, Lei},
          booktitle={IEEE Conference on Computer Vision and Pattern Recognition},
          pages={1671--1681},
          year={2019}
        }
    '''
    x = bicubic_degradation(x, sf=sf)
    x = ndimage.filters.convolve(x, np.expand_dims(k, axis=2), mode='wrap')
    return x


def modcrop_np(img, sf):
    '''
    Args:
        img: numpy image, WxH or WxHxC
        sf: scale factor

    Return:
        cropped image
    '''
    w, h = img.shape[:2]
    im = np.copy(img)
    return im[:w - w % sf, :h - h % sf, ...]


if __name__ == '__main__':
    img = util.imread_uint('test.bmp', 3)

#    img = util.uint2single(img)
#    k = utils_deblur.fspecial('gaussian', 7, 1.6)
#
#    for sf in [2, 3, 4]:
#
#        # modcrop
#        img = modcrop_np(img, sf=sf)
#
#        # 1) bicubic degradation
#        img_b = bicubic_degradation(img, sf=sf)
#        print(img_b.shape)
#
#        # 2) srmd degradation
#        img_s = srmd_degradation(img, k, sf=sf)
#        print(img_s.shape)
#
#        # 3) dpsr degradation
#        img_d = dpsr_degradation(img, k, sf=sf)
#        print(img_d.shape)


#    k = anisotropic_Gaussian(ksize=7, theta=0.25*np.pi, l1=0.01, l2=0.01)
#    print(k)
#    util.imshow(k*10)

#    k = gen_kernel(k_size=np.array([15, 15]), scale_factor=np.array([4, 4]), min_var=0.8, max_var=10.8, noise_level=0.0)
#    util.imshow(k*10)

#    util.surf(k)

    # PCA
#    pca_matrix = cal_pca_matrix(ksize=15, l_max=10.0, dim_pca=15, num_samples=12500)
#    print(pca_matrix.shape)
#    show_pca(pca_matrix)
    # run utils/utils_sisr.py
