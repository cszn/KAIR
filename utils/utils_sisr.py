# -*- coding: utf-8 -*-
from utils import utils_image as util
import random

import scipy
import scipy.stats as ss
import scipy.io as io
from scipy import ndimage
from scipy.interpolate import interp2d

import numpy as np
import torch


"""
# --------------------------------------------
# Super-Resolution
# --------------------------------------------
#
# Kai Zhang (cskaizhang@gmail.com)
# https://github.com/cszn
# modified by Kai Zhang (github: https://github.com/cszn)
# 03/03/2020
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
# shifted anisotropic Gaussian kernels
# --------------------------------------------
"""


def shifted_anisotropic_Gaussian(k_size=np.array([15, 15]), scale_factor=np.array([4, 4]), min_var=0.6, max_var=10., noise_level=0):
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


def gen_kernel(k_size=np.array([25, 25]), scale_factor=np.array([4, 4]), min_var=0.6, max_var=12., noise_level=0):
    """"
    # modified version of https://github.com/assafshocher/BlindSR_dataset_generator
    # Kai Zhang
    # min_var = 0.175 * sf  # variance of the gaussian kernel will be sampled between min_var and max_var
    # max_var = 2.5 * sf
    """
    sf = random.choice([1, 2, 3, 4])
    scale_factor = np.array([sf, sf])
    # Set random eigen-vals (lambdas) and angle (theta) for COV matrix
    lambda_1 = min_var + np.random.rand() * (max_var - min_var)
    lambda_2 = min_var + np.random.rand() * (max_var - min_var)
    theta = np.random.rand() * np.pi  # random theta
    noise = 0#-noise_level + np.random.rand(*k_size) * noise_level * 2

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


def classical_degradation(x, k, sf=3):
    ''' blur + downsampling

    Args:
        x: HxWxC image, [0, 1]/[0, 255]
        k: hxw, double
        sf: down-scale factor

    Return:
        downsampled LR image
    '''
    x = ndimage.filters.convolve(x, np.expand_dims(k, axis=2), mode='wrap')
    #x = filters.correlate(x, np.expand_dims(np.flip(k), axis=2))
    st = 0
    return x[st::sf, st::sf, ...]


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


'''
# =================
# Numpy
# =================
'''


def shift_pixel(x, sf, upper_left=True):
    """shift pixel for super-resolution with different scale factors
    Args:
        x: WxHxC or WxH, image or kernel
        sf: scale factor
        upper_left: shift direction
    """
    h, w = x.shape[:2]
    shift = (sf-1)*0.5
    xv, yv = np.arange(0, w, 1.0), np.arange(0, h, 1.0)
    if upper_left:
        x1 = xv + shift
        y1 = yv + shift
    else:
        x1 = xv - shift
        y1 = yv - shift

    x1 = np.clip(x1, 0, w-1)
    y1 = np.clip(y1, 0, h-1)

    if x.ndim == 2:
        x = interp2d(xv, yv, x)(x1, y1)
    if x.ndim == 3:
        for i in range(x.shape[-1]):
            x[:, :, i] = interp2d(xv, yv, x[:, :, i])(x1, y1)

    return x


'''
# =================
# pytorch
# =================
'''


def splits(a, sf):
    '''
    a: tensor NxCxWxHx2
    sf: scale factor
    out: tensor NxCx(W/sf)x(H/sf)x2x(sf^2)
    '''
    b = torch.stack(torch.chunk(a, sf, dim=2), dim=5)
    b = torch.cat(torch.chunk(b, sf, dim=3), dim=5)
    return b


def c2c(x):
    return torch.from_numpy(np.stack([np.float32(x.real), np.float32(x.imag)], axis=-1))


def r2c(x):
    return torch.stack([x, torch.zeros_like(x)], -1)


def cdiv(x, y):
    a, b = x[..., 0], x[..., 1]
    c, d = y[..., 0], y[..., 1]
    cd2 = c**2 + d**2
    return torch.stack([(a*c+b*d)/cd2, (b*c-a*d)/cd2], -1)


def csum(x, y):
    return torch.stack([x[..., 0] + y, x[..., 1]], -1)


def cabs(x):
    return torch.pow(x[..., 0]**2+x[..., 1]**2, 0.5)


def cmul(t1, t2):
    '''
    complex multiplication
    t1: NxCxHxWx2
    output: NxCxHxWx2
    '''
    real1, imag1 = t1[..., 0], t1[..., 1]
    real2, imag2 = t2[..., 0], t2[..., 1]
    return torch.stack([real1 * real2 - imag1 * imag2, real1 * imag2 + imag1 * real2], dim=-1)


def cconj(t, inplace=False):
    '''
    # complex's conjugation
    t: NxCxHxWx2
    output: NxCxHxWx2
    '''
    c = t.clone() if not inplace else t
    c[..., 1] *= -1
    return c


def rfft(t):
    return torch.rfft(t, 2, onesided=False)


def irfft(t):
    return torch.irfft(t, 2, onesided=False)


def fft(t):
    return torch.fft(t, 2)


def ifft(t):
    return torch.ifft(t, 2)


def p2o(psf, shape):
    '''
    Args:
        psf: NxCxhxw
        shape: [H,W]

    Returns:
        otf: NxCxHxWx2
    '''
    otf = torch.zeros(psf.shape[:-2] + shape).type_as(psf)
    otf[...,:psf.shape[2],:psf.shape[3]].copy_(psf)
    for axis, axis_size in enumerate(psf.shape[2:]):
        otf = torch.roll(otf, -int(axis_size / 2), dims=axis+2)
    otf = torch.rfft(otf, 2, onesided=False)
    n_ops = torch.sum(torch.tensor(psf.shape).type_as(psf) * torch.log2(torch.tensor(psf.shape).type_as(psf)))
    otf[...,1][torch.abs(otf[...,1])<n_ops*2.22e-16] = torch.tensor(0).type_as(psf)
    return otf


'''
# =================
PyTorch
# =================
'''

def INVLS_pytorch(FB, FBC, F2B, FR, tau, sf=2):
    '''
    FB: NxCxWxHx2
    F2B: NxCxWxHx2

    x1 = FB.*FR;
    FBR = BlockMM(nr,nc,Nb,m,x1);
    invW = BlockMM(nr,nc,Nb,m,F2B);
    invWBR = FBR./(invW + tau*Nb);
    fun = @(block_struct) block_struct.data.*invWBR;
    FCBinvWBR = blockproc(FBC,[nr,nc],fun);
    FX = (FR-FCBinvWBR)/tau;
    Xest = real(ifft2(FX));
    '''
    x1 = cmul(FB, FR)
    FBR = torch.mean(splits(x1, sf), dim=-1, keepdim=False)
    invW = torch.mean(splits(F2B, sf), dim=-1, keepdim=False)
    invWBR = cdiv(FBR, csum(invW, tau))
    FCBinvWBR = cmul(FBC, invWBR.repeat(1,1,sf,sf,1))
    FX = (FR-FCBinvWBR)/tau
    Xest = torch.irfft(FX, 2, onesided=False)
    return Xest


def real2complex(x):
    return torch.stack([x, torch.zeros_like(x)], -1)


def modcrop(img, sf):
    '''
    img: tensor image, NxCxWxH or CxWxH or WxH
    sf: scale factor
    '''
    w, h = img.shape[-2:]
    im = img.clone()
    return im[..., :w - w % sf, :h - h % sf]


def upsample(x, sf=3, center=False):
    '''
    x: tensor image, NxCxWxH
    '''
    st = (sf-1)//2 if center else 0
    z = torch.zeros((x.shape[0], x.shape[1], x.shape[2]*sf, x.shape[3]*sf)).type_as(x)
    z[..., st::sf, st::sf].copy_(x)
    return z


def downsample(x, sf=3, center=False):
    st = (sf-1)//2 if center else 0
    return x[..., st::sf, st::sf]


def circular_pad(x, pad):
    '''
    # x[N, 1, W, H] -> x[N, 1, W + 2 pad, H + 2 pad] (pariodic padding)
    '''
    x = torch.cat([x, x[:, :, 0:pad, :]], dim=2)
    x = torch.cat([x, x[:, :, :, 0:pad]], dim=3)
    x = torch.cat([x[:, :, -2 * pad:-pad, :], x], dim=2)
    x = torch.cat([x[:, :, :, -2 * pad:-pad], x], dim=3)
    return x


def pad_circular(input, padding):
    # type: (Tensor, List[int]) -> Tensor
    """
    Arguments
    :param input: tensor of shape :math:`(N, C_{\text{in}}, H, [W, D]))`
    :param padding: (tuple): m-elem tuple where m is the degree of convolution
    Returns
    :return: tensor of shape :math:`(N, C_{\text{in}}, [D + 2 * padding[0],
                                     H + 2 * padding[1]], W + 2 * padding[2]))`
    """
    offset = 3
    for dimension in range(input.dim() - offset + 1):
        input = dim_pad_circular(input, padding[dimension], dimension + offset)
    return input


def dim_pad_circular(input, padding, dimension):
    # type: (Tensor, int, int) -> Tensor
    input = torch.cat([input, input[[slice(None)] * (dimension - 1) +
                      [slice(0, padding)]]], dim=dimension - 1)
    input = torch.cat([input[[slice(None)] * (dimension - 1) +
                      [slice(-2 * padding, -padding)]], input], dim=dimension - 1)
    return input


def imfilter(x, k):
    '''
    x: image, NxcxHxW
    k: kernel, cx1xhxw
    '''
    x = pad_circular(x, padding=((k.shape[-2]-1)//2, (k.shape[-1]-1)//2))
    x = torch.nn.functional.conv2d(x, k, groups=x.shape[1])
    return x


def G(x, k, sf=3, center=False):
    '''
    x: image, NxcxHxW
    k: kernel, cx1xhxw
    sf: scale factor
    center: the first one or the moddle one

    Matlab function:
    tmp = imfilter(x,h,'circular');
    y = downsample2(tmp,K);
    '''
    x = downsample(imfilter(x, k), sf=sf, center=center)
    return x


def Gt(x, k, sf=3, center=False):
    '''
    x: image, NxcxHxW
    k: kernel, cx1xhxw
    sf: scale factor
    center: the first one or the moddle one

    Matlab function:
    tmp = upsample2(x,K);
    y = imfilter(tmp,h,'circular');
    '''
    x = imfilter(upsample(x, sf=sf, center=center), k)
    return x


def interpolation_down(x, sf, center=False):
    mask = torch.zeros_like(x)
    if center:
        start = torch.tensor((sf-1)//2)
        mask[..., start::sf, start::sf] = torch.tensor(1).type_as(x)
        LR = x[..., start::sf, start::sf]
    else:
        mask[..., ::sf, ::sf] = torch.tensor(1).type_as(x)
        LR = x[..., ::sf, ::sf]
    y = x.mul(mask)

    return LR, y, mask


'''
# =================
Numpy
# =================
'''


def blockproc(im, blocksize, fun):
    xblocks = np.split(im, range(blocksize[0], im.shape[0], blocksize[0]), axis=0)
    xblocks_proc = []
    for xb in xblocks:
        yblocks = np.split(xb, range(blocksize[1], im.shape[1], blocksize[1]), axis=1)
        yblocks_proc = []
        for yb in yblocks:
            yb_proc = fun(yb)
            yblocks_proc.append(yb_proc)
        xblocks_proc.append(np.concatenate(yblocks_proc, axis=1))

    proc = np.concatenate(xblocks_proc, axis=0)

    return proc


def fun_reshape(a):
    return np.reshape(a, (-1,1,a.shape[-1]), order='F')


def fun_mul(a, b):
    return a*b


def BlockMM(nr, nc, Nb, m, x1):
    '''
    myfun = @(block_struct) reshape(block_struct.data,m,1);
    x1 = blockproc(x1,[nr nc],myfun);
    x1 = reshape(x1,m,Nb);
    x1 = sum(x1,2);
    x = reshape(x1,nr,nc);
    '''
    fun = fun_reshape
    x1 = blockproc(x1, blocksize=(nr, nc), fun=fun)
    x1 = np.reshape(x1, (m, Nb, x1.shape[-1]), order='F')
    x1 = np.sum(x1, 1)
    x = np.reshape(x1, (nr, nc, x1.shape[-1]), order='F')
    return x


def INVLS(FB, FBC, F2B, FR, tau, Nb, nr, nc, m):
    '''
    x1 = FB.*FR;
    FBR = BlockMM(nr,nc,Nb,m,x1);
    invW = BlockMM(nr,nc,Nb,m,F2B);
    invWBR = FBR./(invW + tau*Nb);
    fun = @(block_struct) block_struct.data.*invWBR;
    FCBinvWBR = blockproc(FBC,[nr,nc],fun);
    FX = (FR-FCBinvWBR)/tau;
    Xest = real(ifft2(FX));
    '''
    x1 = FB*FR
    FBR = BlockMM(nr, nc, Nb, m, x1)
    invW = BlockMM(nr, nc, Nb, m, F2B)
    invWBR = FBR/(invW + tau*Nb)
    FCBinvWBR = blockproc(FBC, [nr, nc], lambda im: fun_mul(im, invWBR))
    FX = (FR-FCBinvWBR)/tau
    Xest = np.real(np.fft.ifft2(FX, axes=(0, 1)))
    return Xest


def psf2otf(psf, shape=None):
    """
    Convert point-spread function to optical transfer function.
    Compute the Fast Fourier Transform (FFT) of the point-spread
    function (PSF) array and creates the optical transfer function (OTF)
    array that is not influenced by the PSF off-centering.
    By default, the OTF array is the same size as the PSF array.
    To ensure that the OTF is not altered due to PSF off-centering, PSF2OTF
    post-pads the PSF array (down or to the right) with zeros to match
    dimensions specified in OUTSIZE, then circularly shifts the values of
    the PSF array up (or to the left) until the central pixel reaches (1,1)
    position.
    Parameters
    ----------
    psf : `numpy.ndarray`
        PSF array
    shape : int
        Output shape of the OTF array
    Returns
    -------
    otf : `numpy.ndarray`
        OTF array
    Notes
    -----
    Adapted from MATLAB psf2otf function
    """
    if type(shape) == type(None):
        shape = psf.shape
    shape = np.array(shape)
    if np.all(psf == 0):
        # return np.zeros_like(psf)
        return np.zeros(shape)
    if len(psf.shape) == 1:
        psf = psf.reshape((1, psf.shape[0]))
    inshape = psf.shape
    psf = zero_pad(psf, shape, position='corner')
    for axis, axis_size in enumerate(inshape):
        psf = np.roll(psf, -int(axis_size / 2), axis=axis)
    # Compute the OTF
    otf = np.fft.fft2(psf, axes=(0, 1))
    # Estimate the rough number of operations involved in the FFT
    # and discard the PSF imaginary part if within roundoff error
    # roundoff error  = machine epsilon = sys.float_info.epsilon
    # or np.finfo().eps
    n_ops = np.sum(psf.size * np.log2(psf.shape))
    otf = np.real_if_close(otf, tol=n_ops)
    return otf


def zero_pad(image, shape, position='corner'):
    """
    Extends image to a certain size with zeros
    Parameters
    ----------
    image: real 2d `numpy.ndarray`
        Input image
    shape: tuple of int
        Desired output shape of the image
    position : str, optional
        The position of the input image in the output one:
            * 'corner'
                top-left corner (default)
            * 'center'
                centered
    Returns
    -------
    padded_img: real `numpy.ndarray`
        The zero-padded image
    """
    shape = np.asarray(shape, dtype=int)
    imshape = np.asarray(image.shape, dtype=int)
    if np.alltrue(imshape == shape):
        return image
    if np.any(shape <= 0):
        raise ValueError("ZERO_PAD: null or negative shape given")
    dshape = shape - imshape
    if np.any(dshape < 0):
        raise ValueError("ZERO_PAD: target size smaller than source one")
    pad_img = np.zeros(shape, dtype=image.dtype)
    idx, idy = np.indices(imshape)
    if position == 'center':
        if np.any(dshape % 2 != 0):
            raise ValueError("ZERO_PAD: source and target shapes "
                             "have different parity.")
        offx, offy = dshape // 2
    else:
        offx, offy = (0, 0)
    pad_img[idx + offx, idy + offy] = image
    return pad_img


def upsample_np(x, sf=3, center=False):
    st = (sf-1)//2 if center else 0
    z = np.zeros((x.shape[0]*sf, x.shape[1]*sf, x.shape[2]))
    z[st::sf, st::sf, ...] = x
    return z


def downsample_np(x, sf=3, center=False):
    st = (sf-1)//2 if center else 0
    return x[st::sf, st::sf, ...]


def imfilter_np(x, k):
    '''
    x: image, NxcxHxW
    k: kernel, cx1xhxw
    '''
    x = ndimage.filters.convolve(x, np.expand_dims(k, axis=2), mode='wrap')
    return x


def G_np(x, k, sf=3, center=False):
    '''
    x: image, NxcxHxW
    k: kernel, cx1xhxw

    Matlab function:
    tmp = imfilter(x,h,'circular');
    y = downsample2(tmp,K);
    '''
    x = downsample_np(imfilter_np(x, k), sf=sf, center=center)
    return x


def Gt_np(x, k, sf=3, center=False):
    '''
    x: image, NxcxHxW
    k: kernel, cx1xhxw

    Matlab function:
    tmp = upsample2(x,K);
    y = imfilter(tmp,h,'circular');
    '''
    x = imfilter_np(upsample_np(x, sf=sf, center=center), k)
    return x


if __name__ == '__main__':
    img = util.imread_uint('test.bmp', 3)

    img = util.uint2single(img)
    k = anisotropic_Gaussian(ksize=15, theta=np.pi, l1=6, l2=6)
    util.imshow(k*10)


    for sf in [2, 3, 4]:

        # modcrop
        img = modcrop_np(img, sf=sf)

        # 1) bicubic degradation
        img_b = bicubic_degradation(img, sf=sf)
        print(img_b.shape)

        # 2) srmd degradation
        img_s = srmd_degradation(img, k, sf=sf)
        print(img_s.shape)

        # 3) dpsr degradation
        img_d = dpsr_degradation(img, k, sf=sf)
        print(img_d.shape)

        # 4) classical degradation
        img_d = classical_degradation(img, k, sf=sf)
        print(img_d.shape)

    k = anisotropic_Gaussian(ksize=7, theta=0.25*np.pi, l1=0.01, l2=0.01)
    #print(k)
#    util.imshow(k*10)

    k = shifted_anisotropic_Gaussian(k_size=np.array([15, 15]), scale_factor=np.array([4, 4]), min_var=0.8, max_var=10.8, noise_level=0.0)
#    util.imshow(k*10)


    # PCA
#    pca_matrix = cal_pca_matrix(ksize=15, l_max=10.0, dim_pca=15, num_samples=12500)
#    print(pca_matrix.shape)
#    show_pca(pca_matrix)
    # run utils/utils_sisr.py
    # run utils_sisr.py
    
    
    
    
    
    
    
