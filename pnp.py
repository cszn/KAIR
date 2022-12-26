import torch
import torch.nn as nn
# import math
import numpy as np
# import matplotlib.pyplot as plt
# from torchvision.transforms import ToPILImage
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
import os.path
from collections import OrderedDict
from copy import deepcopy
from itertools import product

from utils import utils_option as option
from utils import utils_image as util
from utils import utils_logger
from data.select_dataset import define_Dataset
from models.select_network import define_G

def get_opt(json_path):
    opt = option.parse(json_path, is_train=True)
    opt = option.dict_to_nonedict(opt)
    return opt

def get_network(opt):
    model = define_G(opt)
    model_path = opt['path']['pretrained_netG']
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    for k, v in model.named_parameters():
        v.requires_grad = False
    return model

def get_test_loader(opt):
    dataset_opt = opt['datasets']['test']
    dataset_opt['sigma_test'] = opt['sigma_test']
    test_set = define_Dataset(dataset_opt)
    test_loader = DataLoader(test_set, batch_size=1,
                             shuffle=False, num_workers=1,
                             drop_last=False, pin_memory=True)
    return test_loader

def gen_logger(opt):
    util.mkdirs((path for key, path in opt['path'].items() if 'pretrained' not in key))
    logger_name = 'pnp'
    utils_logger.logger_info(logger_name, os.path.join(opt['path']['log'], logger_name+'.log'))
    logger = logging.getLogger(logger_name)
    return logger

class PNP_ADMM(nn.Module):
    def __init__(self, model, pnp_args):
        super(PNP_ADMM, self).__init__()
        self.model = model

        self.lamb = pnp_args['lamb']
        self.sigma2 = pnp_args['sigma2']
        self.denoisor_sigma = pnp_args['denoisor_sigma']
        self.irl1_iter_num = pnp_args['irl1_iter_num']
        self.eps = pnp_args['eps']
        self.admm_iter_num = pnp_args['admm_iter_num']
        self.mu = pnp_args['mu']

        self.max_psnr = 0.
        self.max_ssim = 0.

    def model_forward(self, data):
        # TODO: denoisor_sigma
        predict = self.model(data['L'])
        return predict

    def IRL1(self, f, u, v, b):
        for j in range(self.irl1_iter_num):
            # TODO: cal v
            pass
        v = u
        return v

    def ADMM(self, f, u, v, b):
        # model_input = f / 255.
        model_input = u / 255.
        u1 = self.model(model_input) * 255.
        b1 = b # self.mu
        v1 = self.IRL1(f, u1, v, b1)
        return u1, v1, b1

    def forward(self, f, origin_img=None):
        f *= 255
        u  = f
        v  = f
        b = torch.zeros(f.shape, device=f.device)

        for k in range(self.admm_iter_num):
            u1, v1, b1 = self.ADMM(f, u, v, b)
            if origin_img:
                self.get_intermediate_results(v, origin_img)

        return u1 / 255.

    def get_intermediate_results(self, v, origin_img): # only test
        pre_i = torch.clamp(v / 255., 0., 1.)
        img_E = util.tensor2uint(pre_i)
        img_H = util.tensor2uint(origin_img)
        psnr = util.calculate_psnr(img_E, img_H, border=0)
        ssim = util.calculate_ssim(img_E, img_H, border=0)
        return psnr, ssim

def evaluate(opt):
    '''Given opt, evaluate on testset'''
    device = 'cuda'
    # test_loader = get_test_loader(opt)
    network = get_network(opt)
    network.to(device)
    logger = gen_logger(opt)
    pnp_admm = PNP_ADMM(network, opt['pnp'])
    # logger.info(option.dict2str(opt))
    # logger.info(network)

    H_paths = opt['datasets']['test']['dataroot_H']
    H_paths = util.get_image_paths(H_paths)
    L_paths = H_paths
    noise_level_img = opt['sigma_test']
    n_channels = opt['n_channels']

    test_results = OrderedDict()
    test_results['psnr'] = []
    test_results['ssim'] = []

    for idx, img in enumerate(L_paths):

        # ------------------------------------
        # (1) img_L
        # ------------------------------------

        img_name, ext = os.path.splitext(os.path.basename(img))
        # logger.info('{:->4d}--> {:>10s}'.format(idx+1, img_name+ext))
        img_L = util.imread_uint(img, n_channels=1)
        img_L = util.uint2single(img_L)

        np.random.seed(seed=0)  # for reproducibility
        img_L += np.random.normal(0, noise_level_img/255., img_L.shape)

        # util.imshow(util.single2uint(img_L), title='Noisy image with noise level {}'.format(noise_level_img)) if show_img else None

        img_L = util.single2tensor4(img_L)
        noise_level_map = torch.ones((1, 1, img_L.size(2), img_L.size(3)), dtype=torch.float).mul_(noise_level_img/255.)
        img_L = torch.cat((img_L, noise_level_map), dim=1)
        img_L = img_L.to(device)

        # ------------------------------------
        # (2) img_E
        # ------------------------------------


        img_E = pnp_admm(img_L)

        img_E = util.tensor2uint(img_E)


        # --------------------------------
        # (3) img_H
        # --------------------------------

        img_H = util.imread_uint(H_paths[idx], n_channels=n_channels)
        img_H = img_H.squeeze()

        # --------------------------------
        # PSNR and SSIM
        # --------------------------------

        psnr = util.calculate_psnr(img_E, img_H, border=0)
        ssim = util.calculate_ssim(img_E, img_H, border=0)
        test_results['psnr'].append(psnr)
        test_results['ssim'].append(ssim)
        logger.info('{:s} - PSNR: {:.2f} dB; SSIM: {:.4f}.'.format(img_name+ext, psnr, ssim))

    ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
    ave_ssim = sum(test_results['ssim']) / len(test_results['ssim'])
    logger.info('Average PSNR/SSIM(RGB) - PSNR: {:.2f} dB; SSIM: {:.4f}'.format(ave_psnr, ave_ssim))
    
    return ave_psnr, ave_ssim

def gen_opts(json_path):
    '''Generate all opt grids to be searched'''
    opts = []
    all_opt = get_opt(json_path)
    pnp_opt = all_opt['pnp']
    for lamb in range(*pnp_opt['lamb']):
        for denoisor_sigma in pnp_opt['denoisor_sigma']:
            opt = deepcopy(all_opt)
            opt['pnp']['lamb'] = lamb
            opt['pnp']['denoisor_sigma'] = denoisor_sigma
            opts.append(opt)
    return opts

def search_args(json_path='options/pnp/pnp_drunet.json'):
    opts = gen_opts(json_path)
    opt_max_psnr = None
    opt_max_ssim = None
    max_psnr = 0.
    max_ssim = 0.
    for opt in opts:
        cur_psnr, cur_ssim = evaluate(opt)
        if cur_psnr > max_psnr:
            max_psnr = cur_psnr
            opt_max_psnr = deepcopy(opt)
        if cur_ssim > max_ssim:
            max_ssim = cur_ssim
            opt_max_ssim = deepcopy(opt)

    print(max_psnr, max_ssim)
    
if __name__ == '__main__':
    search_args()