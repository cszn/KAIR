import random
import numpy as np
import torch
import torch.utils.data as data
import utils.utils_image as util
from utils import utils_sisr


import hdf5storage
import os


class DatasetSRMD(data.Dataset):
    '''
    # -----------------------------------------
    # Get L/H/M for noisy image SR with Gaussian kernels.
    # Only "paths_H" is needed, sythesize bicubicly downsampled L on-the-fly.
    # -----------------------------------------
    # e.g., SRMD, H = f(L, kernel, sigma), sigma is noise level
    # -----------------------------------------
    '''

    def __init__(self, opt):
        super(DatasetSRMD, self).__init__()
        self.opt = opt
        self.n_channels = opt['n_channels'] if opt['n_channels'] else 3
        self.sf = opt['scale'] if opt['scale'] else 4
        self.patch_size = self.opt['H_size'] if self.opt['H_size'] else 96
        self.L_size = self.patch_size // self.sf
        self.sigma = opt['sigma'] if opt['sigma'] else [0, 50]
        self.sigma_min, self.sigma_max = self.sigma[0], self.sigma[1]
        self.sigma_test = opt['sigma_test'] if opt['sigma_test'] else 0

        # -------------------------------------
        # PCA projection matrix
        # -------------------------------------
        self.p = hdf5storage.loadmat(os.path.join('kernels', 'srmd_pca_pytorch.mat'))['p']
        self.ksize = int(np.sqrt(self.p.shape[-1]))  # kernel size

        # ------------------------------------
        # get paths of L/H
        # ------------------------------------
        self.paths_H = util.get_image_paths(opt['dataroot_H'])
        self.paths_L = util.get_image_paths(opt['dataroot_L'])

    def __getitem__(self, index):

        # ------------------------------------
        # get H image
        # ------------------------------------
        H_path = self.paths_H[index]
        img_H = util.imread_uint(H_path, self.n_channels)
        img_H = util.uint2single(img_H)

        # ------------------------------------
        # modcrop for SR
        # ------------------------------------
        img_H = util.modcrop(img_H, self.sf)

        # ------------------------------------
        # kernel
        # ------------------------------------
        if self.opt['phase'] == 'train':
            l_max = 10
            theta = np.pi*np.random.rand(1)
            l1 = 0.1+l_max*np.random.rand(1)
            l2 = 0.1+(l1-0.1)*np.random.rand(1)

            kernel = utils_sisr.anisotropic_Gaussian(ksize=self.ksize, theta=theta[0], l1=l1[0], l2=l2[0])
        else:
            kernel = utils_sisr.anisotropic_Gaussian(ksize=self.ksize, theta=np.pi, l1=0.1, l2=0.1)

        k = np.reshape(kernel, (-1), order="F")
        k_reduced = np.dot(self.p, k)
        k_reduced = torch.from_numpy(k_reduced).float()

        # ------------------------------------
        # sythesize L image via specified degradation model
        # ------------------------------------
        H, W, _ = img_H.shape
        img_L = utils_sisr.srmd_degradation(img_H, kernel, self.sf)
        img_L = np.float32(img_L)

        if self.opt['phase'] == 'train':
            """
            # --------------------------------
            # get L/H patch pairs
            # --------------------------------
            """
            H, W, C = img_L.shape

            # --------------------------------
            # randomly crop L patch
            # --------------------------------
            rnd_h = random.randint(0, max(0, H - self.L_size))
            rnd_w = random.randint(0, max(0, W - self.L_size))
            img_L = img_L[rnd_h:rnd_h + self.L_size, rnd_w:rnd_w + self.L_size, :]

            # --------------------------------
            # crop corresponding H patch
            # --------------------------------
            rnd_h_H, rnd_w_H = int(rnd_h * self.sf), int(rnd_w * self.sf)
            img_H = img_H[rnd_h_H:rnd_h_H + self.patch_size, rnd_w_H:rnd_w_H + self.patch_size, :]

            # --------------------------------
            # augmentation - flip and/or rotate
            # --------------------------------
            mode = np.random.randint(0, 8)
            img_L, img_H = util.augment_img(img_L, mode=mode), util.augment_img(img_H, mode=mode)

            # --------------------------------
            # get patch pairs
            # --------------------------------
            img_H, img_L = util.single2tensor3(img_H), util.single2tensor3(img_L)

            # --------------------------------
            # select noise level and get Gaussian noise
            # --------------------------------
            if random.random() < 0.1:
                noise_level = torch.zeros(1).float()
            else:
                noise_level = torch.FloatTensor([np.random.uniform(self.sigma_min, self.sigma_max)])/255.0
                # noise_level = torch.rand(1)*50/255.0
                # noise_level = torch.min(torch.from_numpy(np.float32([7*np.random.chisquare(2.5)/255.0])),torch.Tensor([50./255.]))
    
        else:

            img_H, img_L = util.single2tensor3(img_H), util.single2tensor3(img_L)
            noise_level = noise_level = torch.FloatTensor([self.sigma_test])

        # ------------------------------------
        # add noise
        # ------------------------------------
        noise = torch.randn(img_L.size()).mul_(noise_level).float()
        img_L.add_(noise)

        # ------------------------------------
        # get degradation map M
        # ------------------------------------
        M_vector = torch.cat((k_reduced, noise_level), 0).unsqueeze(1).unsqueeze(1)
        M = M_vector.repeat(1, img_L.size()[-2], img_L.size()[-1])

        """
        # -------------------------------------
        # concat L and noise level map M
        # -------------------------------------
        """

        img_L = torch.cat((img_L, M), 0)
        L_path = H_path

        return {'L': img_L, 'H': img_H, 'L_path': L_path, 'H_path': H_path}

    def __len__(self):
        return len(self.paths_H)
