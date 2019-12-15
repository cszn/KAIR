import os.path
import logging
import re

import numpy as np
from collections import OrderedDict
from scipy.io import loadmat

import torch

from utils import utils_deblur
from utils import utils_sisr as sr
from utils import utils_logger
from utils import utils_image as util
from utils import utils_model


'''
Spyder (Python 3.6)
PyTorch 1.1.0
Windows 10 or Linux

Kai Zhang (cskaizhang@gmail.com)
github: https://github.com/cszn/KAIR
        https://github.com/cszn/SRMD

@inproceedings{zhang2018learning,
  title={Learning a single convolutional super-resolution network for multiple degradations},
  author={Zhang, Kai and Zuo, Wangmeng and Zhang, Lei},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition},
  pages={3262--3271},
  year={2018}
}

% If you have any question, please feel free to contact with me.
% Kai Zhang (e-mail: cskaizhang@gmail.com; github: https://github.com/cszn)

by Kai Zhang (12/Dec./2019)
'''

"""
# --------------------------------------------
|--model_zoo             # model_zoo
   |--srmdnf_x2          # model_name, for noise-free LR image SR
   |--srmdnf_x3 
   |--srmdnf_x4
   |--srmd_x2            # model_name, for noisy LR image
   |--srmd_x3 
   |--srmd_x4
|--testset               # testsets
   |--set5               # testset_name
   |--srbsd68
|--results               # results
   |--set5_srmdnf_x2     # result_name = testset_name + '_' + model_name
   |--set5_srmdnf_x3
   |--set5_srmdnf_x4
   |--set5_srmd_x2
   |--srbsd68_srmd_x2
# --------------------------------------------
"""


def main():

    # ----------------------------------------
    # Preparation
    # ----------------------------------------

    noise_level_img = 0                  # default: 0, noise level for LR image
    noise_level_model = noise_level_img  # noise level for model 
    model_name = 'srmdnf_x4'             # 'srmd_x2' | 'srmd_x3' | 'srmd_x4' | 'srmdnf_x2' | 'srmdnf_x3' | 'srmdnf_x4'
    testset_name = 'set5'                # test set,  'set5' | 'srbsd68'
    sf = [int(s) for s in re.findall(r'\d+', model_name)][0]  # scale factor
    x8 = False                           # default: False, x8 to boost performance
    need_degradation = True              # default: True, use degradation model to generate LR image
    show_img = False                     # default: False




    srmd_pca_path = os.path.join('kernels', 'srmd_pca_matlab.mat')
    task_current = 'sr'       # 'dn' for denoising | 'sr' for super-resolution
    n_channels = 3            # fixed
    in_nc = 18 if 'nf' in model_name else 19
    nc = 128                  # fixed, number of channels
    nb = 12                   # fixed, number of conv layers
    model_pool = 'model_zoo'  # fixed
    testsets = 'testsets'     # fixed
    results = 'results'       # fixed
    result_name = testset_name + '_' + model_name
    border = sf if task_current == 'sr' else 0     # shave boader to calculate PSNR and SSIM
    model_path = os.path.join(model_pool, model_name+'.pth')

    # ----------------------------------------
    # L_path, E_path, H_path
    # ----------------------------------------

    L_path = os.path.join(testsets, testset_name) # L_path, for Low-quality images
    H_path = L_path                               # H_path, for High-quality images
    E_path = os.path.join(results, result_name)   # E_path, for Estimated images
    util.mkdir(E_path)

    if H_path == L_path:
        need_degradation = True
    logger_name = result_name
    utils_logger.logger_info(logger_name, log_path=os.path.join(E_path, logger_name+'.log'))
    logger = logging.getLogger(logger_name)

    need_H = True if H_path is not None else False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ----------------------------------------
    # load model
    # ----------------------------------------

    from models.network_srmd import SRMD as net
    model = net(in_nc=in_nc, out_nc=n_channels, nc=nc, nb=nb, upscale=sf, act_mode='R', upsample_mode='pixelshuffle')
    model.load_state_dict(torch.load(model_path), strict=False)
    model.eval()
    for k, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)
    logger.info('Model path: {:s}'.format(model_path))
    number_parameters = sum(map(lambda x: x.numel(), model.parameters()))
    logger.info('Params number: {}'.format(number_parameters))

    test_results = OrderedDict()
    test_results['psnr'] = []
    test_results['ssim'] = []
    test_results['psnr_y'] = []
    test_results['ssim_y'] = []

    logger.info('model_name:{}, model sigma:{}, image sigma:{}'.format(model_name, noise_level_img, noise_level_model))
    logger.info(L_path)
    L_paths = util.get_image_paths(L_path)
    H_paths = util.get_image_paths(H_path) if need_H else None

    # ----------------------------------------
    # kernel and PCA reduced feature
    # ----------------------------------------

    # kernel = sr.anisotropic_Gaussian(ksize=15, theta=np.pi, l1=4, l2=4)
    kernel = utils_deblur.fspecial('gaussian', 15, 0.01)  # Gaussian kernel, delta kernel 0.01

    P = loadmat(srmd_pca_path)['P']
    degradation_vector = np.dot(P, np.reshape(kernel, (-1), order="F"))
    if 'nf' not in model_name:  # noise-free SR
        degradation_vector = np.append(degradation_vector, noise_level_model/255.)
    degradation_vector = torch.from_numpy(degradation_vector).view(1, -1, 1, 1).float()

    for idx, img in enumerate(L_paths):

        # ------------------------------------
        # (1) img_L
        # ------------------------------------

        img_name, ext = os.path.splitext(os.path.basename(img))
        # logger.info('{:->4d}--> {:>10s}'.format(idx+1, img_name+ext))
        img_L = util.imread_uint(img, n_channels=n_channels)
        img_L = util.uint2single(img_L)

        # degradation process, blur + bicubic downsampling + Gaussian noise
        if need_degradation:
            img_L = util.modcrop(img_L, sf)
            img_L = sr.srmd_degradation(img_L, kernel, sf)  # equivalent to bicubic degradation if kernel is a delta kernel
            np.random.seed(seed=0)  # for reproducibility
            img_L += np.random.normal(0, noise_level_img/255., img_L.shape)

        util.imshow(util.single2uint(img_L), title='LR image with noise level {}'.format(noise_level_img)) if show_img else None

        img_L = util.single2tensor4(img_L)
        degradation_map = degradation_vector.repeat(1, 1, img_L.size(-2), img_L.size(-1))
        img_L = torch.cat((img_L, degradation_map), dim=1)
        img_L = img_L.to(device)

        # ------------------------------------
        # (2) img_E
        # ------------------------------------

        if not x8:
            img_E = model(img_L)
        else:
            img_E = utils_model.test_mode(model, img_L, mode=3, sf=sf)

        img_E = util.tensor2uint(img_E)

        if need_H:

            # --------------------------------
            # (3) img_H
            # --------------------------------

            img_H = util.imread_uint(H_paths[idx], n_channels=n_channels)
            img_H = img_H.squeeze()
            img_H = util.modcrop(img_H, sf)

            # --------------------------------
            # PSNR and SSIM
            # --------------------------------

            psnr = util.calculate_psnr(img_E, img_H, border=border)
            ssim = util.calculate_ssim(img_E, img_H, border=border)
            test_results['psnr'].append(psnr)
            test_results['ssim'].append(ssim)
            logger.info('{:s} - PSNR: {:.2f} dB; SSIM: {:.4f}.'.format(img_name+ext, psnr, ssim))
            util.imshow(np.concatenate([img_E, img_H], axis=1), title='Recovered / Ground-truth') if show_img else None

            if np.ndim(img_H) == 3:  # RGB image
                img_E_y = util.rgb2ycbcr(img_E, only_y=True)
                img_H_y = util.rgb2ycbcr(img_H, only_y=True)
                psnr_y = util.calculate_psnr(img_E_y, img_H_y, border=border)
                ssim_y = util.calculate_ssim(img_E_y, img_H_y, border=border)
                test_results['psnr_y'].append(psnr_y)
                test_results['ssim_y'].append(ssim_y)

        # ------------------------------------
        # save results
        # ------------------------------------

        util.imsave(img_E, os.path.join(E_path, img_name+'.png'))

    if need_H:
        ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
        ave_ssim = sum(test_results['ssim']) / len(test_results['ssim'])
        logger.info('Average PSNR/SSIM(RGB) - {} - x{} --PSNR: {:.2f} dB; SSIM: {:.4f}'.format(result_name, sf, ave_psnr, ave_ssim))
        if np.ndim(img_H) == 3:
            ave_psnr_y = sum(test_results['psnr_y']) / len(test_results['psnr_y'])
            ave_ssim_y = sum(test_results['ssim_y']) / len(test_results['ssim_y'])
            logger.info('Average PSNR/SSIM( Y ) - {} - x{} - PSNR: {:.2f} dB; SSIM: {:.4f}'.format(result_name, sf, ave_psnr_y, ave_ssim_y))

if __name__ == '__main__':

    main()
