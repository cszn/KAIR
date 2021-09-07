import os.path
import random
import numpy as np
import torch
import cv2
import torch.utils.data as data
import utils.utils_image as util


class DatasetDeblocking(data.Dataset):
    """
    # -----------------------------------------------------------------------------------------
    # Get L/H for JPEG compression artifact reduction (deblocking) with fixed quality factor.
    # Only dataroot_H is needed.
    # -----------------------------------------------------------------------------------------
    # e.g., DRUNet, SwinIR
    # -----------------------------------------------------------------------------------------
    """

    def __init__(self, opt):
        super(DatasetDeblocking, self).__init__()
        print('Dataset: JPEG compression artifact reduction (deblocking) with quality factor. Only dataroot_H is needed.')
        self.opt = opt
        self.n_channels = opt['n_channels'] if opt['n_channels'] else 3
        self.patch_size = opt['H_size'] if opt['H_size'] else 64
        self.quality = opt['quality'] if opt['quality'] else 40
        self.quality_test = opt['quality_test'] if opt['quality_test'] else self.quality

        # ------------------------------------
        # get path of H
        # return None if input is None
        # ------------------------------------
        self.paths_H = util.get_image_paths(opt['dataroot_H'])

    def __getitem__(self, index):

        # ------------------------------------
        # get H image
        # ------------------------------------
        H_path = self.paths_H[index]
        img_H = util.imread_uint(H_path, self.n_channels)

        L_path = H_path

        if self.opt['phase'] == 'train':
            """
            # --------------------------------
            # get L/H patch pairs
            # --------------------------------
            """
            H, W, _ = img_H.shape

            # --------------------------------
            # randomly crop the patch
            # --------------------------------
            rnd_h = random.randint(0, max(0, H - self.patch_size))
            rnd_w = random.randint(0, max(0, W - self.patch_size))
            patch_H = img_H[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]

            # --------------------------------
            # augmentation - flip, rotate
            # --------------------------------
            mode = random.randint(0, 7)
            patch_H = util.augment_img(patch_H, mode=mode)

            # --------------------------------
            # add JPEG compression
            # --------------------------------
            result, encimg = cv2.imencode('.jpg', cv2.cvtColor(patch_H, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), self.quality])
            img_L = cv2.cvtColor(cv2.imdecode(encimg, 0 if self.n_channels == 1 else 1), cv2.COLOR_BGR2RGB)

            # --------------------------------
            # HWC to CHW, numpy(uint) to tensor
            # --------------------------------
            img_H = util.uint2tensor3(patch_H)
            img_L = util.uint2tensor3(img_L)

        else:
            """
            # --------------------------------
            # get L/H image pairs
            # --------------------------------
            """

            # --------------------------------
            # add JPEG compression
            # --------------------------------
            result, encimg = cv2.imencode('.jpg', cv2.cvtColor(img_H, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), self.quality_test])
            img_L = cv2.cvtColor(cv2.imdecode(encimg, 0 if self.n_channels == 1 else 1), cv2.COLOR_BGR2RGB)

            # --------------------------------
            # HWC to CHW, numpy to tensor
            # --------------------------------
            img_L = util.uint2tensor3(img_L)
            img_H = util.uint2tensor3(img_H)

        return {'L': img_L, 'H': img_H, 'H_path': H_path, 'L_path': L_path}

    def __len__(self):
        return len(self.paths_H)
