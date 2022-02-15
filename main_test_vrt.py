# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


import argparse
import cv2
import glob
import os
import torch
import requests
import numpy as np
from os import path as osp
from collections import OrderedDict
from torch.utils.data import DataLoader

from models.network_vrt import VRT as net
from utils import utils_image as util
from data.dataset_video_test import VideoRecurrentTestDataset, VideoTestVimeo90KDataset, SingleVideoRecurrentTestDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='001_VRT_videosr_bi_REDS_6frames', help='tasks: 001 to 008')
    parser.add_argument('--sigma', type=int, default=0, help='noise level for denoising: 10, 20, 30, 40, 50')
    parser.add_argument('--folder_lq', type=str, default='testsets/REDS4/sharp_bicubic',
                        help='input low-quality test video folder')
    parser.add_argument('--folder_gt', type=str, default=None,
                        help='input ground-truth test video folder')
    parser.add_argument('--tile', type=int, nargs='+', default=[40,128,128],
                        help='Tile size, [0,0,0] for no tile during testing (testing as a whole)')
    parser.add_argument('--tile_overlap', type=int, nargs='+', default=[2,20,20],
                        help='Overlapping of different tiles')
    args = parser.parse_args()

    # define model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = prepare_model_dataset(args)
    model.eval()
    model = model.to(device)
    if 'vimeo' in args.folder_lq.lower():
        test_set = VideoTestVimeo90KDataset({'dataroot_gt':args.folder_gt, 'dataroot_lq':args.folder_lq,
                                           'meta_info_file': "data/meta_info/meta_info_Vimeo90K_test_GT.txt",
                                            'pad_sequence': True, 'num_frame': 7, 'cache_data': False})
    elif args.folder_gt is not None:
        test_set = VideoRecurrentTestDataset({'dataroot_gt':args.folder_gt, 'dataroot_lq':args.folder_lq,
                                              'sigma':args.sigma, 'num_frame':-1, 'cache_data': False})
    else:
        test_set = SingleVideoRecurrentTestDataset({'dataroot_gt':args.folder_gt, 'dataroot_lq':args.folder_lq,
                                              'sigma':args.sigma, 'num_frame':-1, 'cache_data': False})

    test_loader = DataLoader(dataset=test_set, num_workers=8, batch_size=1, shuffle=False)

    save_dir = f'results/{args.task}'
    os.makedirs(save_dir, exist_ok=True)
    test_results = OrderedDict()
    test_results['psnr'] = []
    test_results['ssim'] = []
    test_results['psnr_y'] = []
    test_results['ssim_y'] = []

    assert len(test_loader) != 0, f'No dataset found at {args.folder_lq}'

    for idx, batch in enumerate(test_loader):
        lq = batch['L'].to(device)
        folder = batch['folder']
        gt = batch['H'] if 'H' in batch else None

        # inference
        with torch.no_grad():
            output = test_video(lq, model, args)

        if 'vimeo' in args.folder_lq.lower():
            output = output[:, 3:4, :, :, :]
            gt = gt.unsqueeze(0)
            batch['lq_path'] = [['im4.png']]

        test_results_folder = OrderedDict()
        test_results_folder['psnr'] = []
        test_results_folder['ssim'] = []
        test_results_folder['psnr_y'] = []
        test_results_folder['ssim_y'] = []

        for i in range(output.shape[1]):
            # save image
            img = output[:, i, ...].data.squeeze().float().cpu().clamp_(0, 1).numpy()
            if img.ndim == 3:
                img = np.transpose(img[[2, 1, 0], :, :], (1, 2, 0))  # CHW-RGB to HCW-BGR
            img = (img * 255.0).round().astype(np.uint8)  # float32 to uint8
            seq_ = osp.basename(batch['lq_path'][i][0]).split('.')[0]
            os.makedirs(f'{save_dir}/{folder[0]}', exist_ok=True)
            cv2.imwrite(f'{save_dir}/{folder[0]}/{seq_}.png', img)

            # evaluate psnr/ssim
            if gt is not None:
                img_gt = gt[:, i, ...].data.squeeze().float().cpu().clamp_(0, 1).numpy()
                if img_gt.ndim == 3:
                    img_gt = np.transpose(img_gt[[2, 1, 0], :, :], (1, 2, 0))  # CHW-RGB to HCW-BGR
                img_gt = (img_gt * 255.0).round().astype(np.uint8)  # float32 to uint8
                img_gt = np.squeeze(img_gt)

                test_results_folder['psnr'].append(util.calculate_psnr(img, img_gt, border=0))
                test_results_folder['ssim'].append(util.calculate_ssim(img, img_gt, border=0))
                if img_gt.ndim == 3:  # RGB image
                    img = util.bgr2ycbcr(img.astype(np.float32) / 255.) * 255.
                    img_gt = util.bgr2ycbcr(img_gt.astype(np.float32) / 255.) * 255.
                    test_results_folder['psnr_y'].append(util.calculate_psnr(img, img_gt, border=0))
                    test_results_folder['ssim_y'].append(util.calculate_ssim(img, img_gt, border=0))
                else:
                    test_results_folder['psnr_y'] = test_results_folder['psnr']
                    test_results_folder['ssim_y'] = test_results_folder['ssim']

        if gt is not None:
            psnr = sum(test_results_folder['psnr']) / len(test_results_folder['psnr'])
            ssim = sum(test_results_folder['ssim']) / len(test_results_folder['ssim'])
            psnr_y = sum(test_results_folder['psnr_y']) / len(test_results_folder['psnr_y'])
            ssim_y = sum(test_results_folder['ssim_y']) / len(test_results_folder['ssim_y'])
            test_results['psnr'].append(psnr)
            test_results['ssim'].append(ssim)
            test_results['psnr_y'].append(psnr_y)
            test_results['ssim_y'].append(ssim_y)
            print('Testing {:20s} ({:2d}/{}) - PSNR: {:.2f} dB; SSIM: {:.4f}; PSNR_Y: {:.2f} dB; SSIM_Y: {:.4f}'.
                      format(folder[0], idx, len(test_loader), psnr, ssim, psnr_y, ssim_y))
        else:
            print('Testing {:20s}  ({:2d}/{})'.format(folder[0], idx, len(test_loader)))

    # summarize psnr/ssim
    if gt is not None:
        ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
        ave_ssim = sum(test_results['ssim']) / len(test_results['ssim'])
        ave_psnr_y = sum(test_results['psnr_y']) / len(test_results['psnr_y'])
        ave_ssim_y = sum(test_results['ssim_y']) / len(test_results['ssim_y'])
        print('\n{} \n-- Average PSNR: {:.2f} dB; SSIM: {:.4f}; PSNR_Y: {:.2f} dB; SSIM_Y: {:.4f}'.
              format(save_dir, ave_psnr, ave_ssim, ave_psnr_y, ave_ssim_y))


def prepare_model_dataset(args):
    ''' prepare model and dataset according to args.task. '''

    # define model
    if args.task == '001_VRT_videosr_bi_REDS_6frames':
        model = net(upscale=4, img_size=[6,64,64], window_size=[6,8,8], depths=[8,8,8,8,8,8,8, 4,4,4,4, 4,4],
                    indep_reconsts=[11,12], embed_dims=[120,120,120,120,120,120,120, 180,180,180,180, 180,180],
                    num_heads=[6,6,6,6,6,6,6, 6,6,6,6, 6,6], pa_frames=2, deformable_groups=12)
        datasets = ['REDS4']
        args.scale = 4
        args.window_size = [6,8,8]
        args.nonblind_denoising = False

    elif args.task == '002_VRT_videosr_bi_REDS_16frames':
        model = net(upscale=4, img_size=[16,64,64], window_size=[8,8,8], depths=[8,8,8,8,8,8,8, 4,4,4,4, 4,4],
                    indep_reconsts=[11,12], embed_dims=[120,120,120,120,120,120,120, 180,180,180,180, 180,180],
                    num_heads=[6,6,6,6,6,6,6, 6,6,6,6, 6,6], pa_frames=6, deformable_groups=24)
        datasets = ['REDS4']
        args.scale = 4
        args.window_size = [8,8,8]
        args.nonblind_denoising = False

    elif args.task in ['003_VRT_videosr_bi_Vimeo_7frames', '004_VRT_videosr_bd_Vimeo_7frames']:
        model = net(upscale=4, img_size=[8,64,64], window_size=[8,8,8], depths=[8,8,8,8,8,8,8, 4,4,4,4, 4,4],
                    indep_reconsts=[11,12], embed_dims=[120,120,120,120,120,120,120, 180,180,180,180, 180,180],
                    num_heads=[6,6,6,6,6,6,6, 6,6,6,6, 6,6], pa_frames=4, deformable_groups=16)
        datasets = ['Vid4'] # 'Vimeo'. Vimeo dataset is too large. Please refer to #training to download it.
        args.scale = 4
        args.window_size = [8,8,8]
        args.nonblind_denoising = False

    elif args.task in ['005_VRT_videodeblurring_DVD']:
        model = net(upscale=1, img_size=[6,192,192], window_size=[6,8,8], depths=[8,8,8,8,8,8,8, 4,4, 4,4],
                    indep_reconsts=[9,10], embed_dims=[96,96,96,96,96,96,96, 120,120, 120,120],
                    num_heads=[6,6,6,6,6,6,6, 6,6, 6,6], pa_frames=2, deformable_groups=16)
        datasets = ['DVD10']
        args.scale = 1
        args.window_size = [6,8,8]
        args.nonblind_denoising = False

    elif args.task in ['006_VRT_videodeblurring_GoPro']:
        model = net(upscale=1, img_size=[6,192,192], window_size=[6,8,8], depths=[8,8,8,8,8,8,8, 4,4, 4,4],
                    indep_reconsts=[9,10], embed_dims=[96,96,96,96,96,96,96, 120,120, 120,120],
                    num_heads=[6,6,6,6,6,6,6, 6,6, 6,6], pa_frames=2, deformable_groups=16)
        datasets = ['GoPro11-part1', 'GoPro11-part2']
        args.scale = 1
        args.window_size = [6,8,8]
        args.nonblind_denoising = False

    elif args.task in ['007_VRT_videodeblurring_REDS']:
        model = net(upscale=1, img_size=[6,192,192], window_size=[6,8,8], depths=[8,8,8,8,8,8,8, 4,4, 4,4],
                    indep_reconsts=[9,10], embed_dims=[96,96,96,96,96,96,96, 120,120, 120,120],
                    num_heads=[6,6,6,6,6,6,6, 6,6, 6,6], pa_frames=2, deformable_groups=16)
        datasets = ['REDS4']
        args.scale = 1
        args.window_size = [6,8,8]
        args.nonblind_denoising = False

    elif args.task == '008_VRT_videodenoising_DAVIS':
        model = net(upscale=1, img_size=[6,192,192], window_size=[6,8,8], depths=[8,8,8,8,8,8,8, 4,4, 4,4],
                    indep_reconsts=[9,10], embed_dims=[96,96,96,96,96,96,96, 120,120, 120,120],
                    num_heads=[6,6,6,6,6,6,6, 6,6, 6,6], pa_frames=2, deformable_groups=16,
                    nonblind_denoising=True)
        datasets = ['Set8', 'DAVIS-test']
        args.scale = 1
        args.window_size = [6,8,8]
        args.nonblind_denoising = True

    # download model
    model_path = f'model_zoo/vrt/{args.task}.pth'
    if os.path.exists(model_path):
        print(f'loading model from ./model_zoo/vrt/{model_path}')
    else:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        url = 'https://github.com/JingyunLiang/VRT/releases/download/v0.0/{}'.format(os.path.basename(model_path))
        r = requests.get(url, allow_redirects=True)
        print(f'downloading model {model_path}')
        open(model_path, 'wb').write(r.content)

    pretrained_model = torch.load(model_path)
    model.load_state_dict(pretrained_model['params'] if 'params' in pretrained_model.keys() else pretrained_model)

    # download datasets
    if os.path.exists(f'{args.folder_lq}'):
        print(f'using dataset from {args.folder_lq}')
    else:
        if 'vimeo' in args.folder_lq.lower():
            print(f'Vimeo dataset is not at {args.folder_lq}! Please refer to #training of Readme.md to download it.')
        else:
            os.makedirs('testsets', exist_ok=True)
            for dataset in datasets:
                url = f'https://github.com/JingyunLiang/VRT/releases/download/v0.0/testset_{dataset}.tar.gz'
                r = requests.get(url, allow_redirects=True)
                print(f'downloading testing dataset {dataset}')
                open(f'testsets/{dataset}.tar.gz', 'wb').write(r.content)
                os.system(f'tar -xvf testsets/{dataset}.tar.gz -C testsets')
                os.system(f'rm testsets/{dataset}.tar.gz')

    return model


def test_video(lq, model, args):
        '''test the video as a whole or as clips (divided temporally). '''

        num_frame_testing = args.tile[0]
        if num_frame_testing:
            # test as multiple clips if out-of-memory
            sf = args.scale
            num_frame_overlapping = args.tile_overlap[0]
            not_overlap_border = False
            b, d, c, h, w = lq.size()
            c = c - 1 if args.nonblind_denoising else c
            stride = num_frame_testing - num_frame_overlapping
            d_idx_list = list(range(0, d-num_frame_testing, stride)) + [max(0, d-num_frame_testing)]
            E = torch.zeros(b, d, c, h*sf, w*sf)
            W = torch.zeros(b, d, 1, 1, 1)

            for d_idx in d_idx_list:
                lq_clip = lq[:, d_idx:d_idx+num_frame_testing, ...]
                out_clip = test_clip(lq_clip, model, args)
                out_clip_mask = torch.ones((b, min(num_frame_testing, d), 1, 1, 1))

                if not_overlap_border:
                    if d_idx < d_idx_list[-1]:
                        out_clip[:, -num_frame_overlapping//2:, ...] *= 0
                        out_clip_mask[:, -num_frame_overlapping//2:, ...] *= 0
                    if d_idx > d_idx_list[0]:
                        out_clip[:, :num_frame_overlapping//2, ...] *= 0
                        out_clip_mask[:, :num_frame_overlapping//2, ...] *= 0

                E[:, d_idx:d_idx+num_frame_testing, ...].add_(out_clip)
                W[:, d_idx:d_idx+num_frame_testing, ...].add_(out_clip_mask)
            output = E.div_(W)
        else:
            # test as one clip (the whole video) if you have enough memory
            window_size = args.window_size
            d_old = lq.size(1)
            d_pad = (window_size[0] - d_old % window_size[0]) % window_size[0]
            lq = torch.cat([lq, torch.flip(lq[:, -d_pad:, ...], [1])], 1) if d_pad else lq
            output = test_clip(lq, model, args)
            output = output[:, :d_old, :, :, :]

        return output


def test_clip(lq, model, args):
    ''' test the clip as a whole or as patches. '''

    sf = args.scale
    window_size = args.window_size
    size_patch_testing = args.tile[1]
    assert size_patch_testing % window_size[-1] == 0, 'testing patch size should be a multiple of window_size.'

    if size_patch_testing:
        # divide the clip to patches (spatially only, tested patch by patch)
        overlap_size = args.tile_overlap[1]
        not_overlap_border = True

        # test patch by patch
        b, d, c, h, w = lq.size()
        c = c - 1 if args.nonblind_denoising else c
        stride = size_patch_testing - overlap_size
        h_idx_list = list(range(0, h-size_patch_testing, stride)) + [max(0, h-size_patch_testing)]
        w_idx_list = list(range(0, w-size_patch_testing, stride)) + [max(0, w-size_patch_testing)]
        E = torch.zeros(b, d, c, h*sf, w*sf)
        W = torch.zeros_like(E)

        for h_idx in h_idx_list:
            for w_idx in w_idx_list:
                in_patch = lq[..., h_idx:h_idx+size_patch_testing, w_idx:w_idx+size_patch_testing]
                out_patch = model(in_patch).detach().cpu()

                out_patch_mask = torch.ones_like(out_patch)

                if not_overlap_border:
                    if h_idx < h_idx_list[-1]:
                        out_patch[..., -overlap_size//2:, :] *= 0
                        out_patch_mask[..., -overlap_size//2:, :] *= 0
                    if w_idx < w_idx_list[-1]:
                        out_patch[..., :, -overlap_size//2:] *= 0
                        out_patch_mask[..., :, -overlap_size//2:] *= 0
                    if h_idx > h_idx_list[0]:
                        out_patch[..., :overlap_size//2, :] *= 0
                        out_patch_mask[..., :overlap_size//2, :] *= 0
                    if w_idx > w_idx_list[0]:
                        out_patch[..., :, :overlap_size//2] *= 0
                        out_patch_mask[..., :, :overlap_size//2] *= 0

                E[..., h_idx*sf:(h_idx+size_patch_testing)*sf, w_idx*sf:(w_idx+size_patch_testing)*sf].add_(out_patch)
                W[..., h_idx*sf:(h_idx+size_patch_testing)*sf, w_idx*sf:(w_idx+size_patch_testing)*sf].add_(out_patch_mask)
        output = E.div_(W)

    else:
        _, _, _, h_old, w_old = lq.size()
        h_pad = (window_size[1] - h_old % window_size[1]) % window_size[1]
        w_pad = (window_size[2] - w_old % window_size[2]) % window_size[2]

        lq = torch.cat([lq, torch.flip(lq[:, :, :, -h_pad:, :], [3])], 3) if h_pad else lq
        lq = torch.cat([lq, torch.flip(lq[:, :, :, :, -w_pad:], [4])], 4) if w_pad else lq

        output = model(lq).detach().cpu()

        output = output[:, :, :, :h_old*sf, :w_old*sf]

    return output


if __name__ == '__main__':
    main()
