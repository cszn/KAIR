import os.path
import logging

import numpy as np
from collections import OrderedDict
from scipy.io import loadmat

import torch

from utils import utils_logger
from utils import utils_model
from utils import utils_image as util


'''
Spyder (Python 3.6)
PyTorch 1.1.0
Windows 10 or Linux

Kai Zhang (cskaizhang@gmail.com)
github: https://github.com/cszn/KAIR
        https://github.com/cszn/DnCNN
        https://github.com/cszn/FFDNet

@article{zhang2018ffdnet,
  title={FFDNet: Toward a fast and flexible solution for CNN-based image denoising},
  author={Zhang, Kai and Zuo, Wangmeng and Zhang, Lei},
  journal={IEEE Transactions on Image Processing},
  volume={27},
  number={9},
  pages={4608--4622},
  year={2018},
  publisher={IEEE}
}

% If you have any question, please feel free to contact with me.
% Kai Zhang (e-mail: cskaizhang@gmail.com; github: https://github.com/cszn)

by Kai Zhang (12/Dec./2019)
'''

"""
# --------------------------------------------
|--model_zoo             # model_zoo
   |--fdncnn_color       # model_name, for color images
   |--fdncnn_gray
   |--fdncnn_color_clip  # for clipped uint8 color images
   |--fdncnn_gray_clip
|--testset               # testsets
   |--set12              # testset_name
   |--bsd68
   |--cbsd68
|--results               # results
   |--set12_fdncnn_color # result_name = testset_name + '_' + model_name
   |--set12_fdncnn_gray
   |--cbsd68_fdncnn_color_clip
# --------------------------------------------
"""


def main():

    # ----------------------------------------
    # Preparation
    # ----------------------------------------

    noise_level_img = 15                 # noise level for noisy image
    noise_level_model = noise_level_img  # noise level for model
    model_name = 'fdncnn_gray'           # 'fdncnn_gray' | 'fdncnn_color' | 'fdncnn_color_clip' | 'fdncnn_gray_clip'
    testset_name = 'bsd68'               # test set,  'bsd68' | 'cbsd68' | 'set12'
    need_degradation = True              # default: True
    x8 = False                           # default: False, x8 to boost performance
    show_img = False                     # default: Falsedefault: False




    task_current = 'dn'       # 'dn' for denoising | 'sr' for super-resolution
    sf = 1                    # unused for denoising
    if 'color' in model_name:
        n_channels = 3        # 3 for color image
    else:
        n_channels = 1        # 1 for grayscale image
    if 'clip' in model_name:
        use_clip = True       # clip the intensities into range of [0, 1]
    else:
        use_clip = False
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

    from models.network_dncnn import FDnCNN as net
    model = net(in_nc=n_channels+1, out_nc=n_channels, nc=64, nb=20, act_mode='R')
    model.load_state_dict(torch.load(model_path), strict=True)
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

        if need_degradation:  # degradation process
            np.random.seed(seed=0)  # for reproducibility
            img_L += np.random.normal(0, noise_level_img/255., img_L.shape)
        if use_clip:
            img_L = util.uint2single(util.single2uint(img_L)) 

        util.imshow(util.single2uint(img_L), title='Noisy image with noise level {}'.format(noise_level_img)) if show_img else None

        img_L = util.single2tensor4(img_L)
        noise_level_map = torch.ones((1, 1, img_L.size(2), img_L.size(3)), dtype=torch.float).mul_(noise_level_model/255.)
        img_L = torch.cat((img_L, noise_level_map), dim=1)
        img_L = img_L.to(device)

        # ------------------------------------
        # (2) img_E
        # ------------------------------------

        if not x8:
            img_E = model(img_L)
        else:
            img_E = utils_model.test_mode(model, img_L, mode=3)

        img_E = util.tensor2uint(img_E)

        if need_H:

            # --------------------------------
            # (3) img_H
            # --------------------------------

            img_H = util.imread_uint(H_paths[idx], n_channels=n_channels)
            img_H = img_H.squeeze()

            # --------------------------------
            # PSNR and SSIM
            # --------------------------------

            psnr = util.calculate_psnr(img_E, img_H, border=border)
            ssim = util.calculate_ssim(img_E, img_H, border=border)
            test_results['psnr'].append(psnr)
            test_results['ssim'].append(ssim)
            logger.info('{:s} - PSNR: {:.2f} dB; SSIM: {:.4f}.'.format(img_name+ext, psnr, ssim))
            util.imshow(np.concatenate([img_E, img_H], axis=1), title='Recovered / Ground-truth') if show_img else None

        # ------------------------------------
        # save results
        # ------------------------------------

        util.imsave(img_E, os.path.join(E_path, img_name+ext))

    if need_H:
        ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
        ave_ssim = sum(test_results['ssim']) / len(test_results['ssim'])
        logger.info('Average PSNR/SSIM(RGB) - {} - PSNR: {:.2f} dB; SSIM: {:.4f}'.format(result_name, ave_psnr, ave_ssim))

if __name__ == '__main__':

    main()
