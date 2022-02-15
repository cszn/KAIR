import glob
import torch
from os import path as osp
import torch.utils.data as data

import utils.utils_video as utils_video


class VideoRecurrentTestDataset(data.Dataset):
    """Video test dataset for recurrent architectures, which takes LR video
    frames as input and output corresponding HR video frames. Modified from
    https://github.com/xinntao/BasicSR/blob/master/basicsr/data/reds_dataset.py

    Supported datasets: Vid4, REDS4, REDSofficial.
    More generally, it supports testing dataset with following structures:

    dataroot
    ├── subfolder1
        ├── frame000
        ├── frame001
        ├── ...
    ├── subfolder1
        ├── frame000
        ├── frame001
        ├── ...
    ├── ...

    For testing datasets, there is no need to prepare LMDB files.

    Args:
        opt (dict): Config for train dataset. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            io_backend (dict): IO backend type and other kwarg.
            cache_data (bool): Whether to cache testing datasets.
            name (str): Dataset name.
            meta_info_file (str): The path to the file storing the list of test
                folders. If not provided, all the folders in the dataroot will
                be used.
            num_frame (int): Window size for input frames.
            padding (str): Padding mode.
    """

    def __init__(self, opt):
        super(VideoRecurrentTestDataset, self).__init__()
        self.opt = opt
        self.cache_data = opt['cache_data']
        self.gt_root, self.lq_root = opt['dataroot_gt'], opt['dataroot_lq']
        self.data_info = {'lq_path': [], 'gt_path': [], 'folder': [], 'idx': [], 'border': []}

        self.imgs_lq, self.imgs_gt = {}, {}
        if 'meta_info_file' in opt:
            with open(opt['meta_info_file'], 'r') as fin:
                subfolders = [line.split(' ')[0] for line in fin]
                subfolders_lq = [osp.join(self.lq_root, key) for key in subfolders]
                subfolders_gt = [osp.join(self.gt_root, key) for key in subfolders]
        else:
            subfolders_lq = sorted(glob.glob(osp.join(self.lq_root, '*')))
            subfolders_gt = sorted(glob.glob(osp.join(self.gt_root, '*')))

        for subfolder_lq, subfolder_gt in zip(subfolders_lq, subfolders_gt):
            # get frame list for lq and gt
            subfolder_name = osp.basename(subfolder_lq)
            img_paths_lq = sorted(list(utils_video.scandir(subfolder_lq, full_path=True)))
            img_paths_gt = sorted(list(utils_video.scandir(subfolder_gt, full_path=True)))

            max_idx = len(img_paths_lq)
            assert max_idx == len(img_paths_gt), (f'Different number of images in lq ({max_idx})'
                                                  f' and gt folders ({len(img_paths_gt)})')

            self.data_info['lq_path'].extend(img_paths_lq)
            self.data_info['gt_path'].extend(img_paths_gt)
            self.data_info['folder'].extend([subfolder_name] * max_idx)
            for i in range(max_idx):
                self.data_info['idx'].append(f'{i}/{max_idx}')
            border_l = [0] * max_idx
            for i in range(self.opt['num_frame'] // 2):
                border_l[i] = 1
                border_l[max_idx - i - 1] = 1
            self.data_info['border'].extend(border_l)

            # cache data or save the frame list
            if self.cache_data:
                print(f'Cache {subfolder_name} for VideoTestDataset...')
                self.imgs_lq[subfolder_name] = utils_video.read_img_seq(img_paths_lq)
                self.imgs_gt[subfolder_name] = utils_video.read_img_seq(img_paths_gt)
            else:
                self.imgs_lq[subfolder_name] = img_paths_lq
                self.imgs_gt[subfolder_name] = img_paths_gt

        # Find unique folder strings
        self.folders = sorted(list(set(self.data_info['folder'])))
        self.sigma = opt['sigma'] / 255. if 'sigma' in opt else 0 # for non-blind video denoising

    def __getitem__(self, index):
        folder = self.folders[index]

        if self.sigma:
        # for non-blind video denoising
            if self.cache_data:
                imgs_gt = self.imgs_gt[folder]
            else:
                imgs_gt = utils_video.read_img_seq(self.imgs_gt[folder])

            torch.manual_seed(0)
            noise_level = torch.ones((1, 1, 1, 1)) * self.sigma
            noise = torch.normal(mean=0, std=noise_level.expand_as(imgs_gt))
            imgs_lq = imgs_gt + noise
            t, _, h, w = imgs_lq.shape
            imgs_lq = torch.cat([imgs_lq, noise_level.expand(t, 1, h, w)], 1)
        else:
        # for video sr and deblurring
            if self.cache_data:
                imgs_lq = self.imgs_lq[folder]
                imgs_gt = self.imgs_gt[folder]
            else:
                imgs_lq = utils_video.read_img_seq(self.imgs_lq[folder])
                imgs_gt = utils_video.read_img_seq(self.imgs_gt[folder])

        return {
            'L': imgs_lq,
            'H': imgs_gt,
            'folder': folder,
            'lq_path': self.imgs_lq[folder],
        }

    def __len__(self):
        return len(self.folders)


class SingleVideoRecurrentTestDataset(data.Dataset):
    """Single ideo test dataset for recurrent architectures, which takes LR video
    frames as input and output corresponding HR video frames (only input LQ path).

    More generally, it supports testing dataset with following structures:

    dataroot
    ├── subfolder1
        ├── frame000
        ├── frame001
        ├── ...
    ├── subfolder1
        ├── frame000
        ├── frame001
        ├── ...
    ├── ...

    For testing datasets, there is no need to prepare LMDB files.

    Args:
        opt (dict): Config for train dataset. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            io_backend (dict): IO backend type and other kwarg.
            cache_data (bool): Whether to cache testing datasets.
            name (str): Dataset name.
            meta_info_file (str): The path to the file storing the list of test
                folders. If not provided, all the folders in the dataroot will
                be used.
            num_frame (int): Window size for input frames.
            padding (str): Padding mode.
    """

    def __init__(self, opt):
        super(SingleVideoRecurrentTestDataset, self).__init__()
        self.opt = opt
        self.cache_data = opt['cache_data']
        self.lq_root = opt['dataroot_lq']
        self.data_info = {'lq_path': [], 'folder': [], 'idx': [], 'border': []}

        self.imgs_lq = {}
        if 'meta_info_file' in opt:
            with open(opt['meta_info_file'], 'r') as fin:
                subfolders = [line.split(' ')[0] for line in fin]
                subfolders_lq = [osp.join(self.lq_root, key) for key in subfolders]
        else:
            subfolders_lq = sorted(glob.glob(osp.join(self.lq_root, '*')))

        for subfolder_lq in subfolders_lq:
            # get frame list for lq and gt
            subfolder_name = osp.basename(subfolder_lq)
            img_paths_lq = sorted(list(utils_video.scandir(subfolder_lq, full_path=True)))

            max_idx = len(img_paths_lq)

            self.data_info['lq_path'].extend(img_paths_lq)
            self.data_info['folder'].extend([subfolder_name] * max_idx)
            for i in range(max_idx):
                self.data_info['idx'].append(f'{i}/{max_idx}')
            border_l = [0] * max_idx
            for i in range(self.opt['num_frame'] // 2):
                border_l[i] = 1
                border_l[max_idx - i - 1] = 1
            self.data_info['border'].extend(border_l)

            # cache data or save the frame list
            if self.cache_data:
                print(f'Cache {subfolder_name} for VideoTestDataset...')
                self.imgs_lq[subfolder_name] = utils_video.read_img_seq(img_paths_lq)
            else:
                self.imgs_lq[subfolder_name] = img_paths_lq

        # Find unique folder strings
        self.folders = sorted(list(set(self.data_info['folder'])))

    def __getitem__(self, index):
        folder = self.folders[index]

        if self.cache_data:
            imgs_lq = self.imgs_lq[folder]
        else:
            imgs_lq = utils_video.read_img_seq(self.imgs_lq[folder])

        return {
            'L': imgs_lq,
            'folder': folder,
            'lq_path': self.imgs_lq[folder],
        }

    def __len__(self):
        return len(self.folders)


class VideoTestVimeo90KDataset(data.Dataset):
    """Video test dataset for Vimeo90k-Test dataset.

    It only keeps the center frame for testing.
    For testing datasets, there is no need to prepare LMDB files.

    Args:
        opt (dict): Config for train dataset. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            io_backend (dict): IO backend type and other kwarg.
            cache_data (bool): Whether to cache testing datasets.
            name (str): Dataset name.
            meta_info_file (str): The path to the file storing the list of test
                folders. If not provided, all the folders in the dataroot will
                be used.
            num_frame (int): Window size for input frames.
            padding (str): Padding mode.
    """

    def __init__(self, opt):
        super(VideoTestVimeo90KDataset, self).__init__()
        self.opt = opt
        self.cache_data = opt['cache_data']
        if self.cache_data:
            raise NotImplementedError('cache_data in Vimeo90K-Test dataset is not implemented.')
        self.gt_root, self.lq_root = opt['dataroot_gt'], opt['dataroot_lq']
        self.data_info = {'lq_path': [], 'gt_path': [], 'folder': [], 'idx': [], 'border': []}
        neighbor_list = [i + (9 - opt['num_frame']) // 2 for i in range(opt['num_frame'])]

        with open(opt['meta_info_file'], 'r') as fin:
            subfolders = [line.split(' ')[0] for line in fin]
        for idx, subfolder in enumerate(subfolders):
            gt_path = osp.join(self.gt_root, subfolder, 'im4.png')
            self.data_info['gt_path'].append(gt_path)
            lq_paths = [osp.join(self.lq_root, subfolder, f'im{i}.png') for i in neighbor_list]
            self.data_info['lq_path'].append(lq_paths)
            self.data_info['folder'].append('vimeo90k')
            self.data_info['idx'].append(f'{idx}/{len(subfolders)}')
            self.data_info['border'].append(0)

        self.pad_sequence = opt.get('pad_sequence', False)

    def __getitem__(self, index):
        lq_path = self.data_info['lq_path'][index]
        gt_path = self.data_info['gt_path'][index]
        imgs_lq = utils_video.read_img_seq(lq_path)
        img_gt = utils_video.read_img_seq([gt_path])
        img_gt.squeeze_(0)

        if self.pad_sequence:  # pad the sequence: 7 frames to 8 frames
            imgs_lq = torch.cat([imgs_lq, imgs_lq[-1:,...]], dim=0)

        return {
            'L': imgs_lq,  # (t, c, h, w)
            'H': img_gt,  # (c, h, w)
            'folder': self.data_info['folder'][index],  # folder name
            'idx': self.data_info['idx'][index],  # e.g., 0/843
            'border': self.data_info['border'][index],  # 0 for non-border
            'lq_path': lq_path[self.opt['num_frame'] // 2]  # center frame
        }

    def __len__(self):
        return len(self.data_info['gt_path'])


class SingleVideoRecurrentTestDataset(data.Dataset):
    """Single Video test dataset (only input LQ path).

    Supported datasets: Vid4, REDS4, REDSofficial.
    More generally, it supports testing dataset with following structures:

    dataroot
    ├── subfolder1
        ├── frame000
        ├── frame001
        ├── ...
    ├── subfolder1
        ├── frame000
        ├── frame001
        ├── ...
    ├── ...

    For testing datasets, there is no need to prepare LMDB files.

    Args:
        opt (dict): Config for train dataset. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            io_backend (dict): IO backend type and other kwarg.
            cache_data (bool): Whether to cache testing datasets.
            name (str): Dataset name.
            meta_info_file (str): The path to the file storing the list of test
                folders. If not provided, all the folders in the dataroot will
                be used.
            num_frame (int): Window size for input frames.
            padding (str): Padding mode.
    """

    def __init__(self, opt):
        super(SingleVideoRecurrentTestDataset, self).__init__()
        self.opt = opt
        self.cache_data = opt['cache_data']
        self.lq_root = opt['dataroot_lq']
        self.data_info = {'lq_path': [], 'folder': [], 'idx': [], 'border': []}
        # file client (io backend)
        self.file_client = None

        self.imgs_lq = {}
        if 'meta_info_file' in opt:
            with open(opt['meta_info_file'], 'r') as fin:
                subfolders = [line.split(' ')[0] for line in fin]
                subfolders_lq = [osp.join(self.lq_root, key) for key in subfolders]
        else:
            subfolders_lq = sorted(glob.glob(osp.join(self.lq_root, '*')))

        for subfolder_lq in subfolders_lq:
            # get frame list for lq and gt
            subfolder_name = osp.basename(subfolder_lq)
            img_paths_lq = sorted(list(utils_video.scandir(subfolder_lq, full_path=True)))

            max_idx = len(img_paths_lq)

            self.data_info['lq_path'].extend(img_paths_lq)
            self.data_info['folder'].extend([subfolder_name] * max_idx)
            for i in range(max_idx):
                self.data_info['idx'].append(f'{i}/{max_idx}')
            border_l = [0] * max_idx
            for i in range(self.opt['num_frame'] // 2):
                border_l[i] = 1
                border_l[max_idx - i - 1] = 1
            self.data_info['border'].extend(border_l)

            # cache data or save the frame list
            if self.cache_data:
                logger.info(f'Cache {subfolder_name} for VideoTestDataset...')
                self.imgs_lq[subfolder_name] = utils_video.read_img_seq(img_paths_lq)
            else:
                self.imgs_lq[subfolder_name] = img_paths_lq

        # Find unique folder strings
        self.folders = sorted(list(set(self.data_info['folder'])))

    def __getitem__(self, index):
        folder = self.folders[index]

        if self.cache_data:
            imgs_lq = self.imgs_lq[folder]
        else:
            imgs_lq = utils_video.read_img_seq(self.imgs_lq[folder])

        return {
            'L': imgs_lq,
            'folder': folder,
            'lq_path': self.imgs_lq[folder],
        }

    def __len__(self):
        return len(self.folders)
