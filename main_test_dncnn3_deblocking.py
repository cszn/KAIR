import os.path
import logging

import numpy as np
from datetime import datetime
from collections import OrderedDict

import torch

from utils import utils_logger
from utils import utils_model
from utils import utils_image as util
#import os
#os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


'''
Spyder (Python 3.6)
PyTorch 1.1.0
Windows 10 or Linux

Kai Zhang (cskaizhang@gmail.com)
github: https://github.com/cszn/KAIR
        https://github.com/cszn/DnCNN

@article{zhang2017beyond,
  title={Beyond a gaussian denoiser: Residual learning of deep cnn for image denoising},
  author={Zhang, Kai and Zuo, Wangmeng and Chen, Yunjin and Meng, Deyu and Zhang, Lei},
  journal={IEEE Transactions on Image Processing},
  volume={26},
  number={7},
  pages={3142--3155},
  year={2017},
  publisher={IEEE}
}

% If you have any question, please feel free to contact with me.
% Kai Zhang (e-mail: cskaizhang@gmail.com; github: https://github.com/cszn)

by Kai Zhang (12/Dec./2019)
'''

"""
# --------------------------------------------
|--model_zoo          # model_zoo
   |--dncnn3          # model_name
|--testset            # testsets
   |--set12           # testset_name
   |--bsd68
|--results            # results
   |--set12_dncnn3    # result_name = testset_name + '_' + model_name
# --------------------------------------------
"""


def main():

    # ----------------------------------------
    # Preparation
    # ----------------------------------------

    model_name = 'dncnn3'     # 'dncnn3'- can be used for blind Gaussian denoising, JPEG deblocking (quality factor 5-100) and super-resolution (x234)

    # important!
    testset_name = 'bsd68'    # test set, low-quality grayscale/color JPEG images
    n_channels = 1            # set 1 for grayscale image, set 3 for color image


    x8 = False                       # default: False, x8 to boost performance
    testsets = 'testsets'     # fixed
    results = 'results'       # fixed
    result_name = testset_name + '_' + model_name # fixed
    L_path = os.path.join(testsets, testset_name) # L_path, for Low-quality grayscale/Y-channel JPEG images
    E_path = os.path.join(results, result_name)   # E_path, for Estimated images
    util.mkdir(E_path)

    model_pool = 'model_zoo'  # fixed
    model_path = os.path.join(model_pool, model_name+'.pth')
    logger_name = result_name
    utils_logger.logger_info(logger_name, log_path=os.path.join(E_path, logger_name+'.log'))
    logger = logging.getLogger(logger_name)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ----------------------------------------
    # load model
    # ----------------------------------------

    from models.network_dncnn import DnCNN as net
    model = net(in_nc=1, out_nc=1, nc=64, nb=20, act_mode='R')
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    for k, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)
    logger.info('Model path: {:s}'.format(model_path))
    number_parameters = sum(map(lambda x: x.numel(), model.parameters()))
    logger.info('Params number: {}'.format(number_parameters))

    logger.info(L_path)
    L_paths = util.get_image_paths(L_path)

    for idx, img in enumerate(L_paths):

        # ------------------------------------
        # (1) img_L
        # ------------------------------------
        img_name, ext = os.path.splitext(os.path.basename(img))
        logger.info('{:->4d}--> {:>10s}'.format(idx+1, img_name+ext))
        img_L = util.imread_uint(img, n_channels=n_channels)
        img_L = util.uint2single(img_L)
        if n_channels == 3:
            ycbcr = util.rgb2ycbcr(img_L, False)
            img_L = ycbcr[..., 0:1]
        img_L = util.single2tensor4(img_L)
        img_L = img_L.to(device)

        # ------------------------------------
        # (2) img_E
        # ------------------------------------
        if not x8:
            img_E = model(img_L)
        else:
            img_E = utils_model.test_mode(model, img_L, mode=3)

        img_E = util.tensor2single(img_E)
        if n_channels == 3:
            ycbcr[..., 0] = img_E
            img_E = util.ycbcr2rgb(ycbcr)
        img_E = util.single2uint(img_E)

        # ------------------------------------
        # save results
        # ------------------------------------
        util.imsave(img_E, os.path.join(E_path, img_name+'.png'))


if __name__ == '__main__':

    main()
