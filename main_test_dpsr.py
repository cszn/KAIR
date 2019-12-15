import os.path
import logging
import re

import numpy as np
from collections import OrderedDict

import torch

from utils import utils_logger
from utils import utils_image as util
from utils import utils_model


'''
Spyder (Python 3.6)
PyTorch 1.1.0
Windows 10 or Linux

Kai Zhang (cskaizhang@gmail.com)
github: https://github.com/cszn/KAIR
        https://github.com/cszn/DPSR

@inproceedings{zhang2019deep,
  title={Deep Plug-and-Play Super-Resolution for Arbitrary Blur Kernels},
  author={Zhang, Kai and Zuo, Wangmeng and Zhang, Lei},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition},
  pages={1671--1681},
  year={2019}
}

% If you have any question, please feel free to contact with me.
% Kai Zhang (e-mail: cskaizhang@gmail.com; github: https://github.com/cszn)

by Kai Zhang (12/Dec./2019)
'''

"""
# --------------------------------------------
testing code for the super-resolver prior of DPSR
# --------------------------------------------
|--model_zoo             # model_zoo
   |--dpsr_x2            # model_name, optimized for PSNR
   |--dpsr_x3 
   |--dpsr_x4
   |--dpsr_x4_gan        # model_name, optimized for perceptual quality
|--testset               # testsets
   |--set5               # testset_name
   |--srbsd68
|--results               # results
   |--set5_dpsr_x2       # result_name = testset_name + '_' + model_name
   |--set5_dpsr_x3
   |--set5_dpsr_x4
   |--set5_dpsr_x4_gan
   |--srbsd68_dpsr_x4_gan
# --------------------------------------------
"""


def main():

    # ----------------------------------------
    # Preparation
    # ----------------------------------------

    noise_level_img = 0                  # default: 0, noise level for LR image
    noise_level_model = noise_level_img  # noise level for model    
    model_name = 'dpsr_x4_gan'           # 'dpsr_x2' | 'dpsr_x3' | 'dpsr_x4' | 'dpsr_x4_gan'
    testset_name = 'set5'                # test set,  'set5' | 'srbsd68'
    need_degradation = True              # default: True
    x8 = False                           # default: False, x8 to boost performance
    sf = [int(s) for s in re.findall(r'\d+', model_name)][0]  # scale factor
    show_img = False                     # default: False



    task_current = 'sr'       # 'dn' for denoising | 'sr' for super-resolution
    n_channels = 3            # fixed
    nc = 96                   # fixed, number of channels
    nb = 16                   # fixed, number of conv layers
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

    from models.network_dpsr import MSRResNet_prior as net
    model = net(in_nc=n_channels+1, out_nc=n_channels, nc=nc, nb=nb, upscale=sf, act_mode='R', upsample_mode='pixelshuffle')
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

    for idx, img in enumerate(L_paths):

        # ------------------------------------
        # (1) img_L
        # ------------------------------------

        img_name, ext = os.path.splitext(os.path.basename(img))
        # logger.info('{:->4d}--> {:>10s}'.format(idx+1, img_name+ext))
        img_L = util.imread_uint(img, n_channels=n_channels)
        img_L = util.uint2single(img_L)

        # degradation process, bicubic downsampling + Gaussian noise
        if need_degradation:
            img_L = util.modcrop(img_L, sf)
            img_L = util.imresize_np(img_L, 1/sf)
            np.random.seed(seed=0)  # for reproducibility
            img_L += np.random.normal(0, noise_level_img/255., img_L.shape)

        util.imshow(util.single2uint(img_L), title='LR image with noise level {}'.format(noise_level_img)) if show_img else None

        img_L = util.single2tensor4(img_L)
        noise_level_map = torch.full((1, 1, img_L.size(2), img_L.size(3)), noise_level_model/255.).type_as(img_L)
        img_L = torch.cat((img_L, noise_level_map), dim=1)
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
