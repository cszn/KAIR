import random
import numpy as np
import torch
import torch.utils.data as data
import utils.utils_image as util


class DatasetDnPatch(data.Dataset):
    """
    # -----------------------------------------
    # Get L/H for denosing on AWGN with fixed sigma.
    # ****Get all H patches first****
    # Only dataroot_H is needed.
    # -----------------------------------------
    # e.g., DnCNN with BSD400
    # -----------------------------------------
    """

    def __init__(self, opt):
        super(DatasetDnPatch, self).__init__()
        print('Get L/H for denosing on AWGN with fixed sigma. Only dataroot_H is needed.')
        self.opt = opt
        self.n_channels = opt['n_channels'] if opt['n_channels'] else 3
        self.patch_size = opt['H_size'] if opt['H_size'] else 64

        self.sigma = opt['sigma'] if opt['sigma'] else 25
        self.sigma_test = opt['sigma_test'] if opt['sigma_test'] else self.sigma

        self.num_patches_per_image = opt['num_patches_per_image'] if opt['num_patches_per_image'] else 40
        self.num_sampled = opt['num_sampled'] if opt['num_sampled'] else 3000

        # ------------------------------------
        # get paths of H
        # ------------------------------------
        self.paths_H = util.get_image_paths(opt['dataroot_H'])
        assert self.paths_H, 'Error: H path is empty.'

        # ------------------------------------
        # number of sampled H images
        # ------------------------------------
        self.num_sampled = min(self.num_sampled, len(self.paths_H))

        # ------------------------------------
        # reserve space with zeros
        # ------------------------------------
        self.total_patches = self.num_sampled * self.num_patches_per_image
        self.H_data = np.zeros([self.total_patches, self.patch_size, self.patch_size, self.n_channels], dtype=np.uint8)

        # ------------------------------------
        # update H patches
        # ------------------------------------
        self.update_data()

    def update_data(self):
        """
        # ------------------------------------
        # update whole H patches
        # ------------------------------------
        """
        self.index_sampled = random.sample(range(0, len(self.paths_H), 1), self.num_sampled)
        n_count = 0

        for i in range(len(self.index_sampled)):
            H_patches = self.get_patches(self.index_sampled[i])
            for H_patch in H_patches:
                self.H_data[n_count,:,:,:] = H_patch
                n_count += 1

        print('Training data updated! Total number of patches is:  %5.2f X %5.2f = %5.2f\n' % (len(self.H_data)//128, 128, len(self.H_data)))

    def get_patches(self, index):
        """
        # ------------------------------------
        # get H patches from an H image
        # ------------------------------------
        """
        H_path = self.paths_H[index]
        img_H = util.imread_uint(H_path, self.n_channels)  # uint format

        H, W = img_H.shape[:2]

        H_patches = []

        num = self.num_patches_per_image
        for _ in range(num):
            rnd_h = random.randint(0, max(0, H - self.patch_size))
            rnd_w = random.randint(0, max(0, W - self.patch_size))
            H_patch = img_H[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]
            H_patches.append(H_patch)

        return H_patches

    def __getitem__(self, index):

        H_path = 'toy.png'
        if self.opt['phase'] == 'train':

            patch_H = self.H_data[index]

            # --------------------------------
            # augmentation - flip and/or rotate
            # --------------------------------
            mode = random.randint(0, 7)
            patch_H = util.augment_img(patch_H, mode=mode)

            patch_H = util.uint2tensor3(patch_H)
            patch_L = patch_H.clone()

            # ------------------------------------
            # add noise
            # ------------------------------------
            noise = torch.randn(patch_L.size()).mul_(self.sigma/255.0)
            patch_L.add_(noise)

        else:

            H_path = self.paths_H[index]
            img_H = util.imread_uint(H_path, self.n_channels)
            img_H = util.uint2single(img_H)
            img_L = np.copy(img_H)

            # ------------------------------------
            # add noise
            # ------------------------------------
            np.random.seed(seed=0)
            img_L += np.random.normal(0, self.sigma_test/255.0, img_L.shape)
            patch_L, patch_H = util.single2tensor3(img_L), util.single2tensor3(img_H)

        L_path = H_path
        return {'L': patch_L, 'H': patch_H, 'L_path': L_path, 'H_path': H_path}

    def __len__(self):
        return len(self.H_data)
