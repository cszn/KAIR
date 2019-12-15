import os.path
import random
import numpy as np
import torch.utils.data as data
import utils.utils_image as util



class DatasetPlainPatch(data.Dataset):
    '''
    # -----------------------------------------
    # Get L/H for image-to-image mapping.
    # Both "paths_L" and "paths_H" are needed.
    # -----------------------------------------
    # e.g., train denoiser with L and H patches
    # create a large patch dataset first
    # -----------------------------------------
    '''

    def __init__(self, opt):
        super(DatasetPlainPatch, self).__init__()
        print('Get L/H for image-to-image mapping. Both "paths_L" and "paths_H" are needed.')
        self.opt = opt
        self.n_channels = opt['n_channels'] if opt['n_channels'] else 3
        self.patch_size = self.opt['H_size'] if self.opt['H_size'] else 64

        self.num_patches_per_image = opt['num_patches_per_image'] if opt['num_patches_per_image'] else 40
        self.num_sampled = opt['num_sampled'] if opt['num_sampled'] else 3000

        # -------------------
        # get the path of L/H
        # -------------------
        self.paths_H = util.get_image_paths(opt['dataroot_H'])
        self.paths_L = util.get_image_paths(opt['dataroot_L'])

        assert self.paths_H, 'Error: H path is empty.'
        assert self.paths_L, 'Error: L path is empty. This dataset uses L path, you can use dataset_dnpatchh'
        if self.paths_L and self.paths_H:
            assert len(self.paths_L) == len(self.paths_H), 'H and L datasets have different number of images - {}, {}.'.format(len(self.paths_L), len(self.paths_H))

        # ------------------------------------
        # number of sampled images
        # ------------------------------------
        self.num_sampled = min(self.num_sampled, len(self.paths_H))

        # ------------------------------------
        # reserve space with zeros
        # ------------------------------------
        self.total_patches = self.num_sampled * self.num_patches_per_image
        self.H_data = np.zeros([self.total_patches, self.path_size, self.path_size, self.n_channels], dtype=np.uint8)
        self.L_data = np.zeros([self.total_patches, self.path_size, self.path_size, self.n_channels], dtype=np.uint8)

        # ------------------------------------
        # update H patches
        # ------------------------------------
        self.update_data()


    def update_data(self):
        """
        # ------------------------------------
        # update whole L/H patches
        # ------------------------------------
        """
        self.index_sampled = random.sample(range(0, len(self.paths_H), 1), self.num_sampled)
        n_count = 0

        for i in range(len(self.index_sampled)):
            L_patches, H_patches = self.get_patches(self.index_sampled[i])
            for (L_patch, H_patch) in zip(L_patches, H_patches):
                self.L_data[n_count,:,:,:] = L_patch
                self.H_data[n_count,:,:,:] = H_patch
                n_count += 1

        print('Training data updated! Total number of patches is:  %5.2f X %5.2f = %5.2f\n' % (len(self.H_data)//128, 128, len(self.H_data)))

    def get_patches(self, index):
        """
        # ------------------------------------
        # get L/H patches from L/H images
        # ------------------------------------
        """
        L_path = self.paths_L[index]
        H_path = self.paths_H[index]
        img_L = util.imread_uint(L_path, self.n_channels)  # uint format
        img_H = util.imread_uint(H_path, self.n_channels)  # uint format

        H, W = img_H.shape[:2]

        L_patches, H_patches = [], []

        num = self.num_patches_per_image
        for _ in range(num):
            rnd_h = random.randint(0, max(0, H - self.path_size))
            rnd_w = random.randint(0, max(0, W - self.path_size))
            L_patch = img_L[rnd_h:rnd_h + self.path_size, rnd_w:rnd_w + self.path_size, :]
            H_patch = img_H[rnd_h:rnd_h + self.path_size, rnd_w:rnd_w + self.path_size, :]
            L_patches.append(L_patch)
            H_patches.append(H_patch)

        return L_patches, H_patches

    def __getitem__(self, index):

        if self.opt['phase'] == 'train':

            patch_L, patch_H = self.L_data[index], self.H_data[index]

            # --------------------------------
            # augmentation - flip and/or rotate
            # --------------------------------
            mode = np.random.randint(0, 8)
            patch_L = util.augment_img(patch_L, mode=mode)
            patch_H = util.augment_img(patch_H, mode=mode)

            patch_L, patch_H = util.uint2tensor3(patch_L), util.uint2tensor3(patch_H)

        else:

            L_path, H_path = self.paths_L[index], self.paths_H[index]
            patch_L = util.imread_uint(L_path, self.n_channels)
            patch_H = util.imread_uint(H_path, self.n_channels)

            patch_L, patch_H = util.uint2tensor3(patch_L), util.uint2tensor3(patch_H)

        return {'L': patch_L, 'H': patch_H}


    def __len__(self):
        
        return self.total_patches
