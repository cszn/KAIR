import sys
import os.path
import math
import argparse
import time
import random
import cv2
import numpy as np
from collections import OrderedDict
import logging
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from utils import utils_logger
from utils import utils_image as util
from utils import utils_option as option
from utils.utils_dist import get_dist_info, init_dist

from data.select_dataset import define_Dataset
from models.select_model import define_Model


'''
# --------------------------------------------
# training code for VRT
# --------------------------------------------
'''


def main(json_path='options/vrt/001_train_vrt_videosr_bi_reds_6frames.json'):

    '''
    # ----------------------------------------
    # Step--1 (prepare opt)
    # ----------------------------------------
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default=json_path, help='Path to option JSON file.')
    parser.add_argument('--launcher', default='pytorch', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--dist', default=False)

    opt = option.parse(parser.parse_args().opt, is_train=True)
    opt['dist'] = parser.parse_args().dist

    # ----------------------------------------
    # distributed settings
    # ----------------------------------------
    if opt['dist']:
        init_dist('pytorch')
    opt['rank'], opt['world_size'] = get_dist_info()
    
    if opt['rank'] == 0:
        util.mkdirs((path for key, path in opt['path'].items() if 'pretrained' not in key))

    # ----------------------------------------
    # update opt
    # ----------------------------------------
    # -->-->-->-->-->-->-->-->-->-->-->-->-->-
    init_iter_G, init_path_G = option.find_last_checkpoint(opt['path']['models'], net_type='G',
                                                           pretrained_path=opt['path']['pretrained_netG'])
    init_iter_E, init_path_E = option.find_last_checkpoint(opt['path']['models'], net_type='E',
                                                           pretrained_path=opt['path']['pretrained_netE'])
    opt['path']['pretrained_netG'] = init_path_G
    opt['path']['pretrained_netE'] = init_path_E
    init_iter_optimizerG, init_path_optimizerG = option.find_last_checkpoint(opt['path']['models'],
                                                                             net_type='optimizerG')
    opt['path']['pretrained_optimizerG'] = init_path_optimizerG
    current_step = max(init_iter_G, init_iter_E, init_iter_optimizerG)

    # --<--<--<--<--<--<--<--<--<--<--<--<--<-

    # ----------------------------------------
    # save opt to  a '../option.json' file
    # ----------------------------------------
    if opt['rank'] == 0:
        option.save(opt)

    # ----------------------------------------
    # return None for missing key
    # ----------------------------------------
    opt = option.dict_to_nonedict(opt)

    # ----------------------------------------
    # configure logger
    # ----------------------------------------
    if opt['rank'] == 0:
        logger_name = 'train'
        utils_logger.logger_info(logger_name, os.path.join(opt['path']['log'], logger_name+'.log'))
        logger = logging.getLogger(logger_name)
        logger.info(option.dict2str(opt))

    # ----------------------------------------
    # seed
    # ----------------------------------------
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    print('Random seed: {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    '''
    # ----------------------------------------
    # Step--2 (creat dataloader)
    # ----------------------------------------
    '''

    # ----------------------------------------
    # 1) create_dataset
    # 2) creat_dataloader for train and test
    # ----------------------------------------
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = define_Dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt['dataloader_batch_size']))
            if opt['rank'] == 0:
                logger.info('Number of train images: {:,d}, iters: {:,d}'.format(len(train_set), train_size))
            if opt['dist']:
                train_sampler = DistributedSampler(train_set, shuffle=dataset_opt['dataloader_shuffle'],
                                                   drop_last=True, seed=seed)
                train_loader = DataLoader(train_set,
                                          batch_size=dataset_opt['dataloader_batch_size']//opt['num_gpu'],
                                          shuffle=False,
                                          num_workers=dataset_opt['dataloader_num_workers']//opt['num_gpu'],
                                          drop_last=True,
                                          pin_memory=True,
                                          sampler=train_sampler)
            else:
                train_loader = DataLoader(train_set,
                                          batch_size=dataset_opt['dataloader_batch_size'],
                                          shuffle=dataset_opt['dataloader_shuffle'],
                                          num_workers=dataset_opt['dataloader_num_workers'],
                                          drop_last=True,
                                          pin_memory=True)

        elif phase == 'test':
            test_set = define_Dataset(dataset_opt)
            test_loader = DataLoader(test_set, batch_size=1,
                                     shuffle=False, num_workers=1,
                                     drop_last=False, pin_memory=True)
        else:
            raise NotImplementedError("Phase [%s] is not recognized." % phase)

    '''
    # ----------------------------------------
    # Step--3 (initialize model)
    # ----------------------------------------
    '''

    model = define_Model(opt)
    model.init_train()
    if opt['rank'] == 0:
        logger.info(model.info_network())
        logger.info(model.info_params())

    '''
    # ----------------------------------------
    # Step--4 (main training)
    # ----------------------------------------
    '''

    for epoch in range(1000000):  # keep running
        for i, train_data in enumerate(train_loader):

            current_step += 1

            # -------------------------------
            # 1) update learning rate
            # -------------------------------
            model.update_learning_rate(current_step)

            # -------------------------------
            # 2) feed patch pairs
            # -------------------------------
            model.feed_data(train_data)

            # -------------------------------
            # 3) optimize parameters
            # -------------------------------
            model.optimize_parameters(current_step)

            # -------------------------------
            # 4) training information
            # -------------------------------
            if current_step % opt['train']['checkpoint_print'] == 0 and opt['rank'] == 0:
                logs = model.current_log()  # such as loss
                message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> '.format(epoch, current_step,
                                                                          model.current_learning_rate())
                for k, v in logs.items():  # merge log information into message
                    message += '{:s}: {:.3e} '.format(k, v)
                logger.info(message)

            # -------------------------------
            # 5) save model
            # -------------------------------
            if current_step % opt['train']['checkpoint_save'] == 0 and opt['rank'] == 0:
                logger.info('Saving the model.')
                model.save(current_step)

            if opt['use_static_graph'] and (current_step == opt['train']['fix_iter'] - 1):
                current_step += 1
                model.update_learning_rate(current_step)
                model.save(current_step)
                current_step -= 1
                logger.info('Saving models ahead of time when changing the computation graph with use_static_graph=True'
                            ' (we need it due to a bug with use_checkpoint=True in distributed training). The training '
                            'will be terminated by PyTorch in the next iteration. Just resume training with the same '
                            '.json config file.')

            # -------------------------------
            # 6) testing
            # -------------------------------
            if current_step % opt['train']['checkpoint_test'] == 0 and opt['rank'] == 0:

                test_results = OrderedDict()
                test_results['psnr'] = []
                test_results['ssim'] = []
                test_results['psnr_y'] = []
                test_results['ssim_y'] = []

                for idx, test_data in enumerate(test_loader):
                    model.feed_data(test_data)
                    model.test()

                    visuals = model.current_visuals()
                    output = visuals['E']
                    gt = visuals['H'] if 'H' in visuals else None
                    folder = test_data['folder']

                    test_results_folder = OrderedDict()
                    test_results_folder['psnr'] = []
                    test_results_folder['ssim'] = []
                    test_results_folder['psnr_y'] = []
                    test_results_folder['ssim_y'] = []

                    for i in range(output.shape[0]):
                        # -----------------------
                        # save estimated image E
                        # -----------------------
                        img = output[i, ...].clamp_(0, 1).numpy()
                        if img.ndim == 3:
                            img = np.transpose(img[[2, 1, 0], :, :], (1, 2, 0))  # CHW-RGB to HCW-BGR
                        img = (img * 255.0).round().astype(np.uint8)  # float32 to uint8
                        if opt['val']['save_img']:
                            save_dir = opt['path']['images']
                            util.mkdir(save_dir)
                            seq_ = os.path.basename(test_data['lq_path'][i][0]).split('.')[0]
                            os.makedirs(f'{save_dir}/{folder[0]}', exist_ok=True)
                            cv2.imwrite(f'{save_dir}/{folder[0]}/{seq_}_{current_step:d}.png', img)

                        # -----------------------
                        # calculate PSNR
                        # -----------------------
                        img_gt = gt[i, ...].clamp_(0, 1).numpy()
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

                    psnr = sum(test_results_folder['psnr']) / len(test_results_folder['psnr'])
                    ssim = sum(test_results_folder['ssim']) / len(test_results_folder['ssim'])
                    psnr_y = sum(test_results_folder['psnr_y']) / len(test_results_folder['psnr_y'])
                    ssim_y = sum(test_results_folder['ssim_y']) / len(test_results_folder['ssim_y'])

                    if gt is not None:
                        logger.info('Testing {:20s} ({:2d}/{}) - PSNR: {:.2f} dB; SSIM: {:.4f}; '
                                    'PSNR_Y: {:.2f} dB; SSIM_Y: {:.4f}'.
                                    format(folder[0], idx, len(test_loader), psnr, ssim, psnr_y, ssim_y))
                        test_results['psnr'].append(psnr)
                        test_results['ssim'].append(ssim)
                        test_results['psnr_y'].append(psnr_y)
                        test_results['ssim_y'].append(ssim_y)
                    else:
                        logger.info('Testing {:20s}  ({:2d}/{})'.format(folder[0], idx, len(test_loader)))

                # summarize psnr/ssim
                if gt is not None:
                    ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
                    ave_ssim = sum(test_results['ssim']) / len(test_results['ssim'])
                    ave_psnr_y = sum(test_results['psnr_y']) / len(test_results['psnr_y'])
                    ave_ssim_y = sum(test_results['ssim_y']) / len(test_results['ssim_y'])
                    logger.info('<epoch:{:3d}, iter:{:8,d} Average PSNR: {:.2f} dB; SSIM: {:.4f}; '
                                'PSNR_Y: {:.2f} dB; SSIM_Y: {:.4f}'.format(
                        epoch, current_step, ave_psnr, ave_ssim, ave_psnr_y, ave_ssim_y))

            if current_step > opt['train']['total_iter']:
                logger.info('Finish training.')
                model.save(current_step)
                sys.exit()

if __name__ == '__main__':
    main()
