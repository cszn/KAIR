import random
import torch.utils.data as data
import utils.utils_image as util
import cv2


class DatasetJPEG(data.Dataset):
    def __init__(self, opt):
        super(DatasetJPEG, self).__init__()
        print('Dataset: JPEG compression artifact reduction (deblocking) with quality factor. Only dataroot_H is needed.')
        self.opt = opt
        self.n_channels = opt['n_channels'] if opt['n_channels'] else 3
        self.patch_size = self.opt['H_size'] if opt['H_size'] else 128

        self.quality_factor = opt['quality_factor'] if opt['quality_factor'] else 40
        self.quality_factor_test = opt['quality_factor_test'] if opt['quality_factor_test'] else 40
        self.is_color = opt['is_color'] if opt['is_color'] else False

        # -------------------------------------
        # get the path of H, return None if input is None
        # -------------------------------------
        self.paths_H = util.get_image_paths(opt['dataroot_H'])

    def __getitem__(self, index):

        if self.opt['phase'] == 'train':
            # -------------------------------------
            # get H image
            # -------------------------------------
            H_path = self.paths_H[index]
            img_H = util.imread_uint(H_path, 3)
            L_path = H_path

            H, W = img_H.shape[:2]
            self.patch_size_plus = self.patch_size + 8

            # ---------------------------------
            # randomly crop a large patch
            # ---------------------------------
            rnd_h = random.randint(0, max(0, H - self.patch_size_plus))
            rnd_w = random.randint(0, max(0, W - self.patch_size_plus))
            patch_H = img_H[rnd_h:rnd_h + self.patch_size_plus, rnd_w:rnd_w + self.patch_size_plus, ...]

            # ---------------------------------
            # augmentation - flip, rotate
            # ---------------------------------
            mode = random.randint(0, 7)
            patch_H = util.augment_img(patch_H, mode=mode)

            # ---------------------------------
            # HWC to CHW, numpy(uint) to tensor
            # ---------------------------------
            img_L = patch_H.copy()

            # ---------------------------------
            # set quality factor
            # ---------------------------------
            quality_factor = self.quality_factor

            if self.is_color:  # color image
                img_H = img_L.copy()
                img_L = cv2.cvtColor(img_L, cv2.COLOR_RGB2BGR)
                result, encimg = cv2.imencode('.jpg', img_L, [int(cv2.IMWRITE_JPEG_QUALITY), quality_factor])
                img_L = cv2.imdecode(encimg, 1)
                img_L = cv2.cvtColor(img_L, cv2.COLOR_BGR2RGB)
            else:
                if random.random() > 0.5:
                    img_L = util.rgb2ycbcr(img_L)
                else:
                    img_L = cv2.cvtColor(img_L, cv2.COLOR_RGB2GRAY)
                img_H = img_L.copy()
                result, encimg = cv2.imencode('.jpg', img_L, [int(cv2.IMWRITE_JPEG_QUALITY), quality_factor])
                img_L = cv2.imdecode(encimg, 0)

            # ---------------------------------
            # randomly crop a patch
            # ---------------------------------
            H, W = img_H.shape[:2]
            if random.random() > 0.5:
                rnd_h = random.randint(0, max(0, H - self.patch_size))
                rnd_w = random.randint(0, max(0, W - self.patch_size))
            else:
                rnd_h = 0
                rnd_w = 0
            img_H = img_H[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size]
            img_L = img_L[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size]
        else:

            H_path = self.paths_H[index]
            L_path = H_path
            # ---------------------------------
            # set quality factor
            # ---------------------------------
            quality_factor = self.quality_factor_test

            if self.is_color:  # color JPEG image deblocking
                img_H = util.imread_uint(H_path, 3)
                img_L = img_H.copy()
                img_L = cv2.cvtColor(img_L, cv2.COLOR_RGB2BGR)
                result, encimg = cv2.imencode('.jpg', img_L, [int(cv2.IMWRITE_JPEG_QUALITY), quality_factor])
                img_L = cv2.imdecode(encimg, 1)
                img_L = cv2.cvtColor(img_L, cv2.COLOR_BGR2RGB)
            else:
                img_H = cv2.imread(H_path, cv2.IMREAD_UNCHANGED)
                is_to_ycbcr = True if img_L.ndim == 3 else False
                if is_to_ycbcr:
                    img_H = cv2.cvtColor(img_H, cv2.COLOR_BGR2RGB)
                    img_H = util.rgb2ycbcr(img_H)

                result, encimg = cv2.imencode('.jpg', img_H, [int(cv2.IMWRITE_JPEG_QUALITY), quality_factor])
                img_L = cv2.imdecode(encimg, 0)

        img_L, img_H = util.uint2tensor3(img_L), util.uint2tensor3(img_H)

        return {'L': img_L, 'H': img_H, 'L_path': L_path, 'H_path': H_path}

    def __len__(self):
        return len(self.paths_H)
