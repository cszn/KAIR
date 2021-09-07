import random

import numpy as np
import torch
import torch.utils.data as data
import utils.utils_image as util
from utils import utils_deblur
from utils import utils_sisr
import os

from scipy import ndimage
from scipy.io import loadmat
# import hdf5storage


class DatasetUSRNet(data.Dataset):
    '''
    # -----------------------------------------
    # Get L/k/sf/sigma for USRNet.
    # Only "paths_H" and kernel is needed, synthesize L on-the-fly.
    # -----------------------------------------
    '''
    def __init__(self, opt):
        super(DatasetUSRNet, self).__init__()
        self.opt = opt
        self.n_channels = opt['n_channels'] if opt['n_channels'] else 3
        self.patch_size = self.opt['H_size'] if self.opt['H_size'] else 96
        self.sigma_max = self.opt['sigma_max'] if self.opt['sigma_max'] is not None else 25
        self.scales = opt['scales'] if opt['scales'] is not None else [1,2,3,4]
        self.sf_validation = opt['sf_validation'] if opt['sf_validation'] is not None else 3
        #self.kernels = hdf5storage.loadmat(os.path.join('kernels', 'kernels_12.mat'))['kernels']
        self.kernels = loadmat(os.path.join('kernels', 'kernels_12.mat'))['kernels']  # for validation

        # -------------------
        # get the path of H
        # -------------------
        self.paths_H = util.get_image_paths(opt['dataroot_H'])  # return None if input is None
        self.count = 0

    def __getitem__(self, index):

        # -------------------
        # get H image
        # -------------------
        H_path = self.paths_H[index]
        img_H = util.imread_uint(H_path, self.n_channels)
        L_path = H_path

        if self.opt['phase'] == 'train':

            # ---------------------------
            # 1) scale factor, ensure each batch only involves one scale factor
            # ---------------------------
            if self.count % self.opt['dataloader_batch_size'] == 0:
                # sf = random.choice([1,2,3,4])
                self.sf = random.choice(self.scales)
                # self.count = 0  # optional
            self.count += 1
            H, W, _ = img_H.shape

            # ----------------------------
            # randomly crop the patch
            # ----------------------------
            rnd_h = random.randint(0, max(0, H - self.patch_size))
            rnd_w = random.randint(0, max(0, W - self.patch_size))
            patch_H = img_H[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]

            # ---------------------------
            # augmentation - flip, rotate
            # ---------------------------
            mode = np.random.randint(0, 8)
            patch_H = util.augment_img(patch_H, mode=mode)

            # ---------------------------
            # 2) kernel
            # ---------------------------
            r_value = random.randint(0, 7)
            if r_value>3:
                k = utils_deblur.blurkernel_synthesis(h=25)  # motion blur
            else:
                sf_k = random.choice(self.scales)
                k = utils_sisr.gen_kernel(scale_factor=np.array([sf_k, sf_k]))  # Gaussian blur
                mode_k = random.randint(0, 7)
                k = util.augment_img(k, mode=mode_k)

            # ---------------------------
            # 3) noise level
            # ---------------------------
            if random.randint(0, 8) == 1:
                noise_level = 0/255.0
            else:
                noise_level = np.random.randint(0, self.sigma_max)/255.0

            # ---------------------------
            # Low-quality image
            # ---------------------------
            img_L = ndimage.filters.convolve(patch_H, np.expand_dims(k, axis=2), mode='wrap')
            img_L = img_L[0::self.sf, 0::self.sf, ...]
            # add Gaussian noise
            img_L = util.uint2single(img_L) + np.random.normal(0, noise_level, img_L.shape)
            img_H = patch_H

        else:

            k = self.kernels[0, 0].astype(np.float64)  # validation kernel
            k /= np.sum(k)
            noise_level = 0./255.0  # validation noise level

            # ------------------------------------
            # modcrop
            # ------------------------------------
            img_H = util.modcrop(img_H, self.sf_validation)

            img_L = ndimage.filters.convolve(img_H, np.expand_dims(k, axis=2), mode='wrap')  # blur
            img_L = img_L[0::self.sf_validation, 0::self.sf_validation, ...]  # downsampling
            img_L = util.uint2single(img_L) + np.random.normal(0, noise_level, img_L.shape)
            self.sf = self.sf_validation

        k = util.single2tensor3(np.expand_dims(np.float32(k), axis=2))
        img_H, img_L = util.uint2tensor3(img_H), util.single2tensor3(img_L)
        noise_level = torch.FloatTensor([noise_level]).view([1,1,1])

        return {'L': img_L, 'H': img_H, 'k': k, 'sigma': noise_level, 'sf': self.sf, 'L_path': L_path, 'H_path': H_path}

    def __len__(self):
        return len(self.paths_H)
