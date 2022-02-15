import numpy as np
import random
import torch
from pathlib import Path
import torch.utils.data as data

import utils.utils_video as utils_video


class VideoRecurrentTrainDataset(data.Dataset):
    """Video dataset for training recurrent networks.

    The keys are generated from a meta info txt file.
    basicsr/data/meta_info/meta_info_XXX_GT.txt

    Each line contains:
    1. subfolder (clip) name; 2. frame number; 3. image shape, separated by
    a white space.
    Examples:
    720p_240fps_1 100 (720,1280,3)
    720p_240fps_3 100 (720,1280,3)
    ...

    Key examples: "720p_240fps_1/00000"
    GT (gt): Ground-Truth;
    LQ (lq): Low-Quality, e.g., low-resolution/blurry/noisy/compressed frames.

    Args:
        opt (dict): Config for train dataset. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            dataroot_flow (str, optional): Data root path for flow.
            meta_info_file (str): Path for meta information file.
            val_partition (str): Validation partition types. 'REDS4' or
                'official'.
            io_backend (dict): IO backend type and other kwarg.

            num_frame (int): Window size for input frames.
            gt_size (int): Cropped patched size for gt patches.
            interval_list (list): Interval list for temporal augmentation.
            random_reverse (bool): Random reverse input frames.
            use_hflip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h
                and w for implementation).

            scale (bool): Scale, which will be added automatically.
    """

    def __init__(self, opt):
        super(VideoRecurrentTrainDataset, self).__init__()
        self.opt = opt
        self.scale = opt.get('scale', 4)
        self.gt_size = opt.get('gt_size', 256)
        self.gt_root, self.lq_root = Path(opt['dataroot_gt']), Path(opt['dataroot_lq'])
        self.filename_tmpl = opt.get('filename_tmpl', '08d')
        self.filename_ext = opt.get('filename_ext', 'png')
        self.num_frame = opt['num_frame']

        keys = []
        total_num_frames = [] # some clips may not have 100 frames
        start_frames = [] # some clips may not start from 00000
        with open(opt['meta_info_file'], 'r') as fin:
            for line in fin:
                folder, frame_num, _, start_frame = line.split(' ')
                keys.extend([f'{folder}/{i:{self.filename_tmpl}}' for i in range(int(start_frame), int(start_frame)+int(frame_num))])
                total_num_frames.extend([int(frame_num) for i in range(int(frame_num))])
                start_frames.extend([int(start_frame) for i in range(int(frame_num))])

        # remove the video clips used in validation
        if opt['name'] == 'REDS':
            if opt['val_partition'] == 'REDS4':
                val_partition = ['000', '011', '015', '020']
            elif opt['val_partition'] == 'official':
                val_partition = [f'{v:03d}' for v in range(240, 270)]
            else:
                raise ValueError(f'Wrong validation partition {opt["val_partition"]}.'
                                 f"Supported ones are ['official', 'REDS4'].")
        else:
            val_partition = []

        self.keys = []
        self.total_num_frames = [] # some clips may not have 100 frames
        self.start_frames = []
        if opt['test_mode']:
            for i, v in zip(range(len(keys)), keys):
                if v.split('/')[0] in val_partition:
                    self.keys.append(keys[i])
                    self.total_num_frames.append(total_num_frames[i])
                    self.start_frames.append(start_frames[i])
        else:
            for i, v in zip(range(len(keys)), keys):
                if v.split('/')[0] not in val_partition:
                    self.keys.append(keys[i])
                    self.total_num_frames.append(total_num_frames[i])
                    self.start_frames.append(start_frames[i])

        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.is_lmdb = False
        if self.io_backend_opt['type'] == 'lmdb':
            self.is_lmdb = True
            if hasattr(self, 'flow_root') and self.flow_root is not None:
                self.io_backend_opt['db_paths'] = [self.lq_root, self.gt_root, self.flow_root]
                self.io_backend_opt['client_keys'] = ['lq', 'gt', 'flow']
            else:
                self.io_backend_opt['db_paths'] = [self.lq_root, self.gt_root]
                self.io_backend_opt['client_keys'] = ['lq', 'gt']

        # temporal augmentation configs
        self.interval_list = opt.get('interval_list', [1])
        self.random_reverse = opt.get('random_reverse', False)
        interval_str = ','.join(str(x) for x in self.interval_list)
        print(f'Temporal augmentation interval list: [{interval_str}]; '
                    f'random reverse is {self.random_reverse}.')

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = utils_video.FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        key = self.keys[index]
        total_num_frames = self.total_num_frames[index]
        start_frames = self.start_frames[index]
        clip_name, frame_name = key.split('/')  # key example: 000/00000000

        # determine the neighboring frames
        interval = random.choice(self.interval_list)

        # ensure not exceeding the borders
        start_frame_idx = int(frame_name)
        endmost_start_frame_idx = start_frames + total_num_frames - self.num_frame * interval
        if start_frame_idx > endmost_start_frame_idx:
            start_frame_idx = random.randint(start_frames, endmost_start_frame_idx)
        end_frame_idx = start_frame_idx + self.num_frame * interval

        neighbor_list = list(range(start_frame_idx, end_frame_idx, interval))

        # random reverse
        if self.random_reverse and random.random() < 0.5:
            neighbor_list.reverse()

        # get the neighboring LQ and GT frames
        img_lqs = []
        img_gts = []
        for neighbor in neighbor_list:
            if self.is_lmdb:
                img_lq_path = f'{clip_name}/{neighbor:{self.filename_tmpl}}'
                img_gt_path = f'{clip_name}/{neighbor:{self.filename_tmpl}}'
            else:
                img_lq_path = self.lq_root / clip_name / f'{neighbor:{self.filename_tmpl}}.{self.filename_ext}'
                img_gt_path = self.gt_root / clip_name / f'{neighbor:{self.filename_tmpl}}.{self.filename_ext}'

            # get LQ
            img_bytes = self.file_client.get(img_lq_path, 'lq')
            img_lq = utils_video.imfrombytes(img_bytes, float32=True)
            img_lqs.append(img_lq)

            # get GT
            img_bytes = self.file_client.get(img_gt_path, 'gt')
            img_gt = utils_video.imfrombytes(img_bytes, float32=True)
            img_gts.append(img_gt)

        # randomly crop
        img_gts, img_lqs = utils_video.paired_random_crop(img_gts, img_lqs, self.gt_size, self.scale, img_gt_path)

        # augmentation - flip, rotate
        img_lqs.extend(img_gts)
        img_results = utils_video.augment(img_lqs, self.opt['use_hflip'], self.opt['use_rot'])

        img_results = utils_video.img2tensor(img_results)
        img_gts = torch.stack(img_results[len(img_lqs) // 2:], dim=0)
        img_lqs = torch.stack(img_results[:len(img_lqs) // 2], dim=0)

        # img_lqs: (t, c, h, w)
        # img_gts: (t, c, h, w)
        # key: str
        return {'L': img_lqs, 'H': img_gts, 'key': key}

    def __len__(self):
        return len(self.keys)


class VideoRecurrentTrainNonblindDenoisingDataset(VideoRecurrentTrainDataset):
    """Video dataset for training recurrent architectures in non-blind video denoising.

    Args:
        Same as VideoTestDataset.

    """

    def __init__(self, opt):
        super(VideoRecurrentTrainNonblindDenoisingDataset, self).__init__(opt)
        self.sigma_min = self.opt['sigma_min'] / 255.
        self.sigma_max = self.opt['sigma_max'] / 255.

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = utils_video.FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        key = self.keys[index]
        total_num_frames = self.total_num_frames[index]
        start_frames = self.start_frames[index]
        clip_name, frame_name = key.split('/')  # key example: 000/00000000

        # determine the neighboring frames
        interval = random.choice(self.interval_list)

        # ensure not exceeding the borders
        start_frame_idx = int(frame_name)
        endmost_start_frame_idx = start_frames + total_num_frames - self.num_frame * interval
        if start_frame_idx > endmost_start_frame_idx:
            start_frame_idx = random.randint(start_frames, endmost_start_frame_idx)
        end_frame_idx = start_frame_idx + self.num_frame * interval

        neighbor_list = list(range(start_frame_idx, end_frame_idx, interval))

        # random reverse
        if self.random_reverse and random.random() < 0.5:
            neighbor_list.reverse()

        # get the neighboring GT frames
        img_gts = []
        for neighbor in neighbor_list:
            if self.is_lmdb:
                img_gt_path = f'{clip_name}/{neighbor:{self.filename_tmpl}}'
            else:
                img_gt_path = self.gt_root / clip_name / f'{neighbor:{self.filename_tmpl}}.{self.filename_ext}'

            # get GT
            img_bytes = self.file_client.get(img_gt_path, 'gt')
            img_gt = utils_video.imfrombytes(img_bytes, float32=True)
            img_gts.append(img_gt)

        # randomly crop
        img_gts, _ = utils_video.paired_random_crop(img_gts, img_gts, self.gt_size, 1, img_gt_path)

        # augmentation - flip, rotate
        img_gts = utils_video.augment(img_gts, self.opt['use_hflip'], self.opt['use_rot'])

        img_gts = utils_video.img2tensor(img_gts)
        img_gts = torch.stack(img_gts, dim=0)

        # we add noise in the network
        noise_level = torch.empty((1, 1, 1, 1)).uniform_(self.sigma_min, self.sigma_max)
        noise = torch.normal(mean=0, std=noise_level.expand_as(img_gts))
        img_lqs = img_gts + noise

        t, _, h, w = img_lqs.shape
        img_lqs = torch.cat([img_lqs, noise_level.expand(t, 1, h, w)], 1)

        # img_lqs: (t, c, h, w)
        # img_gts: (t, c, h, w)
        # key: str
        return {'L': img_lqs, 'H': img_gts, 'key': key}


    def __len__(self):
        return len(self.keys)


class VideoRecurrentTrainVimeoDataset(data.Dataset):
    """Vimeo90K dataset for training recurrent networks.

    The keys are generated from a meta info txt file.
    basicsr/data/meta_info/meta_info_Vimeo90K_train_GT.txt

    Each line contains:
    1. clip name; 2. frame number; 3. image shape, separated by a white space.
    Examples:
        00001/0001 7 (256,448,3)
        00001/0002 7 (256,448,3)

    Key examples: "00001/0001"
    GT (gt): Ground-Truth;
    LQ (lq): Low-Quality, e.g., low-resolution/blurry/noisy/compressed frames.

    The neighboring frame list for different num_frame:
    num_frame | frame list
             1 | 4
             3 | 3,4,5
             5 | 2,3,4,5,6
             7 | 1,2,3,4,5,6,7

    Args:
        opt (dict): Config for train dataset. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.

            num_frame (int): Window size for input frames.
            gt_size (int): Cropped patched size for gt patches.
            random_reverse (bool): Random reverse input frames.
            use_hflip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h
                and w for implementation).

            scale (bool): Scale, which will be added automatically.
    """

    def __init__(self, opt):
        super(VideoRecurrentTrainVimeoDataset, self).__init__()
        self.opt = opt
        self.gt_root, self.lq_root = Path(opt['dataroot_gt']), Path(opt['dataroot_lq'])

        with open(opt['meta_info_file'], 'r') as fin:
            self.keys = [line.split(' ')[0] for line in fin]

        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.is_lmdb = False
        if self.io_backend_opt['type'] == 'lmdb':
            self.is_lmdb = True
            self.io_backend_opt['db_paths'] = [self.lq_root, self.gt_root]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']

        # indices of input images
        self.neighbor_list = [i + (9 - opt['num_frame']) // 2 for i in range(opt['num_frame'])]

        # temporal augmentation configs
        self.random_reverse = opt['random_reverse']
        print(f'Random reverse is {self.random_reverse}.')

        self.flip_sequence = opt.get('flip_sequence', False)
        self.pad_sequence = opt.get('pad_sequence', False)
        self.neighbor_list = [1, 2, 3, 4, 5, 6, 7]

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = utils_video.FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        # random reverse
        if self.random_reverse and random.random() < 0.5:
            self.neighbor_list.reverse()

        scale = self.opt['scale']
        gt_size = self.opt['gt_size']
        key = self.keys[index]
        clip, seq = key.split('/')  # key example: 00001/0001

        # get the neighboring LQ and  GT frames
        img_lqs = []
        img_gts = []
        for neighbor in self.neighbor_list:
            if self.is_lmdb:
                img_lq_path = f'{clip}/{seq}/im{neighbor}'
                img_gt_path = f'{clip}/{seq}/im{neighbor}'
            else:
                img_lq_path = self.lq_root / clip / seq / f'im{neighbor}.png'
                img_gt_path = self.gt_root / clip / seq / f'im{neighbor}.png'
            # LQ
            img_bytes = self.file_client.get(img_lq_path, 'lq')
            img_lq = utils_video.imfrombytes(img_bytes, float32=True)
            # GT
            img_bytes = self.file_client.get(img_gt_path, 'gt')
            img_gt = utils_video.imfrombytes(img_bytes, float32=True)

            img_lqs.append(img_lq)
            img_gts.append(img_gt)

        # randomly crop
        img_gts, img_lqs = utils_video.paired_random_crop(img_gts, img_lqs, gt_size, scale, img_gt_path)

        # augmentation - flip, rotate
        img_lqs.extend(img_gts)
        img_results = utils_video.augment(img_lqs, self.opt['use_hflip'], self.opt['use_rot'])

        img_results = utils_video.img2tensor(img_results)
        img_lqs = torch.stack(img_results[:7], dim=0)
        img_gts = torch.stack(img_results[7:], dim=0)

        if self.flip_sequence:  # flip the sequence: 7 frames to 14 frames
            img_lqs = torch.cat([img_lqs, img_lqs.flip(0)], dim=0)
            img_gts = torch.cat([img_gts, img_gts.flip(0)], dim=0)
        elif self.pad_sequence:  # pad the sequence: 7 frames to 8 frames
            img_lqs = torch.cat([img_lqs, img_lqs[-1:,...]], dim=0)
            img_gts = torch.cat([img_gts, img_gts[-1:,...]], dim=0)

        # img_lqs: (t, c, h, w)
        # img_gt: (c, h, w)
        # key: str
        return {'L': img_lqs, 'H': img_gts, 'key': key}

    def __len__(self):
        return len(self.keys)
