from collections import OrderedDict
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.optim import Adam

from models.select_network import define_G
from models.model_plain import ModelPlain
from models.loss import CharbonnierLoss
from models.loss_ssim import SSIMLoss

from utils.utils_model import test_mode
from utils.utils_regularizers import regularizer_orth, regularizer_clip


class ModelVRT(ModelPlain):
    """Train video restoration  with pixel loss"""
    def __init__(self, opt):
        super(ModelVRT, self).__init__(opt)
        self.fix_iter = self.opt_train.get('fix_iter', 0)
        self.fix_keys = self.opt_train.get('fix_keys', [])
        self.fix_unflagged = True

    # ----------------------------------------
    # define optimizer
    # ----------------------------------------
    def define_optimizer(self):
        self.fix_keys = self.opt_train.get('fix_keys', [])
        if self.opt_train.get('fix_iter', 0) and len(self.fix_keys) > 0:
            fix_lr_mul = self.opt_train['fix_lr_mul']
            print(f'Multiple the learning rate for keys: {self.fix_keys} with {fix_lr_mul}.')
            if fix_lr_mul == 1:
                G_optim_params = self.netG.parameters()
            else:  # separate flow params and normal params for different lr
                normal_params = []
                flow_params = []
                for name, param in self.netG.named_parameters():
                    if any([key in name for key in self.fix_keys]):
                        flow_params.append(param)
                    else:
                        normal_params.append(param)
                G_optim_params = [
                    {  # add normal params first
                        'params': normal_params,
                        'lr': self.opt_train['G_optimizer_lr']
                    },
                    {
                        'params': flow_params,
                        'lr': self.opt_train['G_optimizer_lr'] * fix_lr_mul
                    },
                ]

            if self.opt_train['G_optimizer_type'] == 'adam':
                self.G_optimizer = Adam(G_optim_params, lr=self.opt_train['G_optimizer_lr'],
                                        betas=self.opt_train['G_optimizer_betas'],
                                        weight_decay=self.opt_train['G_optimizer_wd'])
            else:
                raise NotImplementedError
        else:
            super(ModelVRT, self).define_optimizer()

    # ----------------------------------------
    # update parameters and get loss
    # ----------------------------------------
    def optimize_parameters(self, current_step):
        if self.fix_iter:
            if self.fix_unflagged and current_step < self.fix_iter:
                print(f'Fix keys: {self.fix_keys} for the first {self.fix_iter} iters.')
                self.fix_unflagged = False
                for name, param in self.netG.named_parameters():
                    if any([key in name for key in self.fix_keys]):
                        param.requires_grad_(False)
            elif current_step == self.fix_iter:
                print(f'Train all the parameters from {self.fix_iter} iters.')
                self.netG.requires_grad_(True)

        super(ModelVRT, self).optimize_parameters(current_step)

    # ----------------------------------------
    # test / inference
    # ----------------------------------------
    def test(self):
        n = self.L.size(1)
        self.netG.eval()

        pad_seq = self.opt_train.get('pad_seq', False)
        flip_seq = self.opt_train.get('flip_seq', False)
        self.center_frame_only = self.opt_train.get('center_frame_only', False)

        if pad_seq:
            n = n + 1
            self.L = torch.cat([self.L, self.L[:, -1:, :, :, :]], dim=1)

        if flip_seq:
            self.L = torch.cat([self.L, self.L.flip(1)], dim=1)

        with torch.no_grad():
            self.E = self._test_video(self.L)

        if flip_seq:
            output_1 = self.E[:, :n, :, :, :]
            output_2 = self.E[:, n:, :, :, :].flip(1)
            self.E = 0.5 * (output_1 + output_2)

        if pad_seq:
            n = n - 1
            self.E = self.E[:, :n, :, :, :]

        if self.center_frame_only:
            self.E = self.E[:, n // 2, :, :, :]

        self.netG.train()

    def _test_video(self, lq):
        '''test the video as a whole or as clips (divided temporally). '''

        num_frame_testing = self.opt['val'].get('num_frame_testing', 0)

        if num_frame_testing:
            # test as multiple clips if out-of-memory
            sf = self.opt['scale']
            num_frame_overlapping = self.opt['val'].get('num_frame_overlapping', 2)
            not_overlap_border = False
            b, d, c, h, w = lq.size()
            c = c - 1 if self.opt['netG'].get('nonblind_denoising', False) else c
            stride = num_frame_testing - num_frame_overlapping
            d_idx_list = list(range(0, d-num_frame_testing, stride)) + [max(0, d-num_frame_testing)]
            E = torch.zeros(b, d, c, h*sf, w*sf)
            W = torch.zeros(b, d, 1, 1, 1)

            for d_idx in d_idx_list:
                lq_clip = lq[:, d_idx:d_idx+num_frame_testing, ...]
                out_clip = self._test_clip(lq_clip)
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
            window_size = self.opt['netG'].get('window_size', [6,8,8])
            d_old = lq.size(1)
            d_pad = (d_old// window_size[0]+1)*window_size[0] - d_old
            lq = torch.cat([lq, torch.flip(lq[:, -d_pad:, ...], [1])], 1)
            output = self._test_clip(lq)
            output = output[:, :d_old, :, :, :]

        return output

    def _test_clip(self, lq):
        ''' test the clip as a whole or as patches. '''

        sf = self.opt['scale']
        window_size = self.opt['netG'].get('window_size', [6,8,8])
        size_patch_testing = self.opt['val'].get('size_patch_testing', 0)
        assert size_patch_testing % window_size[-1] == 0, 'testing patch size should be a multiple of window_size.'

        if size_patch_testing:
            # divide the clip to patches (spatially only, tested patch by patch)
            overlap_size = 20
            not_overlap_border = True

            # test patch by patch
            b, d, c, h, w = lq.size()
            c = c - 1 if self.opt['netG'].get('nonblind_denoising', False) else c
            stride = size_patch_testing - overlap_size
            h_idx_list = list(range(0, h-size_patch_testing, stride)) + [max(0, h-size_patch_testing)]
            w_idx_list = list(range(0, w-size_patch_testing, stride)) + [max(0, w-size_patch_testing)]
            E = torch.zeros(b, d, c, h*sf, w*sf)
            W = torch.zeros_like(E)

            for h_idx in h_idx_list:
                for w_idx in w_idx_list:
                    in_patch = lq[..., h_idx:h_idx+size_patch_testing, w_idx:w_idx+size_patch_testing]
                    if hasattr(self, 'netE'):
                        out_patch = self.netE(in_patch).detach().cpu()
                    else:
                        out_patch = self.netG(in_patch).detach().cpu()

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
            h_pad = (h_old// window_size[1]+1)*window_size[1] - h_old
            w_pad = (w_old// window_size[2]+1)*window_size[2] - w_old

            lq = torch.cat([lq, torch.flip(lq[:, :, :, -h_pad:, :], [3])], 3)
            lq = torch.cat([lq, torch.flip(lq[:, :, :, :, -w_pad:], [4])], 4)

            if hasattr(self, 'netE'):
                output = self.netE(lq).detach().cpu()
            else:
                output = self.netG(lq).detach().cpu()

            output = output[:, :, :, :h_old*sf, :w_old*sf]

        return output

    # ----------------------------------------
    # load the state_dict of the network
    # ----------------------------------------
    def load_network(self, load_path, network, strict=True, param_key='params'):
        network = self.get_bare_model(network)
        state_dict = torch.load(load_path)
        if param_key in state_dict.keys():
            state_dict = state_dict[param_key]
        self._print_different_keys_loading(network, state_dict, strict)
        network.load_state_dict(state_dict, strict=strict)

    def _print_different_keys_loading(self, crt_net, load_net, strict=True):
        crt_net = self.get_bare_model(crt_net)
        crt_net = crt_net.state_dict()
        crt_net_keys = set(crt_net.keys())
        load_net_keys = set(load_net.keys())

        if crt_net_keys != load_net_keys:
            print('Current net - loaded net:')
            for v in sorted(list(crt_net_keys - load_net_keys)):
                print(f'  {v}')
            print('Loaded net - current net:')
            for v in sorted(list(load_net_keys - crt_net_keys)):
                print(f'  {v}')

        # check the size for the same keys
        if not strict:
            common_keys = crt_net_keys & load_net_keys
            for k in common_keys:
                if crt_net[k].size() != load_net[k].size():
                    print(f'Size different, ignore [{k}]: crt_net: '
                                   f'{crt_net[k].shape}; load_net: {load_net[k].shape}')
                    load_net[k + '.ignore'] = load_net.pop(k)

