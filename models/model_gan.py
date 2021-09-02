from collections import OrderedDict
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.optim import Adam

from models.select_network import define_G, define_D
from models.model_base import ModelBase
from models.loss import GANLoss, PerceptualLoss
from models.loss_ssim import SSIMLoss


class ModelGAN(ModelBase):
    """Train with pixel-VGG-GAN loss"""
    def __init__(self, opt):
        super(ModelGAN, self).__init__(opt)
        # ------------------------------------
        # define network
        # ------------------------------------
        self.opt_train = self.opt['train']    # training option
        self.netG = define_G(opt)
        self.netG = self.model_to_device(self.netG)
        if self.is_train:
            self.netD = define_D(opt)
            self.netD = self.model_to_device(self.netD)
            if self.opt_train['E_decay'] > 0:
                self.netE = define_G(opt).to(self.device).eval()

    """
    # ----------------------------------------
    # Preparation before training with data
    # Save model during training
    # ----------------------------------------
    """

    # ----------------------------------------
    # initialize training
    # ----------------------------------------
    def init_train(self):
        self.load()                           # load model
        self.netG.train()                     # set training mode,for BN
        self.netD.train()                     # set training mode,for BN
        self.define_loss()                    # define loss
        self.define_optimizer()               # define optimizer
        self.load_optimizers()                # load optimizer
        self.define_scheduler()               # define scheduler
        self.log_dict = OrderedDict()         # log

    # ----------------------------------------
    # load pre-trained G and D model
    # ----------------------------------------
    def load(self):
        load_path_G = self.opt['path']['pretrained_netG']
        if load_path_G is not None:
            print('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG, strict=self.opt_train['G_param_strict'])
        load_path_E = self.opt['path']['pretrained_netE']
        if self.opt_train['E_decay'] > 0:
            if load_path_E is not None:
                print('Loading model for E [{:s}] ...'.format(load_path_E))
                self.load_network(load_path_E, self.netE, strict=self.opt_train['E_param_strict'])
            else:
                print('Copying model for E')
                self.update_E(0)
            self.netE.eval()

        load_path_D = self.opt['path']['pretrained_netD']
        if self.opt['is_train'] and load_path_D is not None:
            print('Loading model for D [{:s}] ...'.format(load_path_D))
            self.load_network(load_path_D, self.netD, strict=self.opt_train['D_param_strict'])

    # ----------------------------------------
    # load optimizerG and optimizerD
    # ----------------------------------------
    def load_optimizers(self):
        load_path_optimizerG = self.opt['path']['pretrained_optimizerG']
        if load_path_optimizerG is not None and self.opt_train['G_optimizer_reuse']:
            print('Loading optimizerG [{:s}] ...'.format(load_path_optimizerG))
            self.load_optimizer(load_path_optimizerG, self.G_optimizer)
        load_path_optimizerD = self.opt['path']['pretrained_optimizerD']
        if load_path_optimizerD is not None and self.opt_train['D_optimizer_reuse']:
            print('Loading optimizerD [{:s}] ...'.format(load_path_optimizerD))
            self.load_optimizer(load_path_optimizerD, self.D_optimizer)

    # ----------------------------------------
    # save model / optimizer(optional)
    # ----------------------------------------
    def save(self, iter_label):
        self.save_network(self.save_dir, self.netG, 'G', iter_label)
        self.save_network(self.save_dir, self.netD, 'D', iter_label)
        if self.opt_train['E_decay'] > 0:
            self.save_network(self.save_dir, self.netE, 'E', iter_label)
        if self.opt_train['G_optimizer_reuse']:
            self.save_optimizer(self.save_dir, self.G_optimizer, 'optimizerG', iter_label)
        if self.opt_train['D_optimizer_reuse']:
            self.save_optimizer(self.save_dir, self.D_optimizer, 'optimizerD', iter_label)

    # ----------------------------------------
    # define loss
    # ----------------------------------------
    def define_loss(self):
        # ------------------------------------
        # 1) G_loss
        # ------------------------------------
        if self.opt_train['G_lossfn_weight'] > 0:
            G_lossfn_type = self.opt_train['G_lossfn_type']
            if G_lossfn_type == 'l1':
                self.G_lossfn = nn.L1Loss().to(self.device)
            elif G_lossfn_type == 'l2':
                self.G_lossfn = nn.MSELoss().to(self.device)
            elif G_lossfn_type == 'l2sum':
                self.G_lossfn = nn.MSELoss(reduction='sum').to(self.device)
            elif G_lossfn_type == 'ssim':
                self.G_lossfn = SSIMLoss().to(self.device)
            else:
                raise NotImplementedError('Loss type [{:s}] is not found.'.format(G_lossfn_type))
            self.G_lossfn_weight = self.opt_train['G_lossfn_weight']
        else:
            print('Do not use pixel loss.')
            self.G_lossfn = None

        # ------------------------------------
        # 2) F_loss
        # ------------------------------------
        if self.opt_train['F_lossfn_weight'] > 0:
            F_feature_layer = self.opt_train['F_feature_layer']
            F_weights = self.opt_train['F_weights']
            F_lossfn_type = self.opt_train['F_lossfn_type']
            F_use_input_norm = self.opt_train['F_use_input_norm']
            F_use_range_norm = self.opt_train['F_use_range_norm']
            if self.opt['dist']:
                self.F_lossfn = PerceptualLoss(feature_layer=F_feature_layer, weights=F_weights, lossfn_type=F_lossfn_type, use_input_norm=F_use_input_norm, use_range_norm=F_use_range_norm).to(self.device)
            else:
                self.F_lossfn = PerceptualLoss(feature_layer=F_feature_layer, weights=F_weights, lossfn_type=F_lossfn_type, use_input_norm=F_use_input_norm, use_range_norm=F_use_range_norm)
                self.F_lossfn.vgg = self.model_to_device(self.F_lossfn.vgg)
                self.F_lossfn.lossfn = self.F_lossfn.lossfn.to(self.device)
            self.F_lossfn_weight = self.opt_train['F_lossfn_weight']
        else:
            print('Do not use feature loss.')
            self.F_lossfn = None

        # ------------------------------------
        # 3) D_loss
        # ------------------------------------
        self.D_lossfn = GANLoss(self.opt_train['gan_type'], 1.0, 0.0).to(self.device)
        self.D_lossfn_weight = self.opt_train['D_lossfn_weight']

        self.D_update_ratio = self.opt_train['D_update_ratio'] if self.opt_train['D_update_ratio'] else 1
        self.D_init_iters = self.opt_train['D_init_iters'] if self.opt_train['D_init_iters'] else 0

    # ----------------------------------------
    # define optimizer, G and D
    # ----------------------------------------
    def define_optimizer(self):
        G_optim_params = []
        for k, v in self.netG.named_parameters():
            if v.requires_grad:
                G_optim_params.append(v)
            else:
                print('Params [{:s}] will not optimize.'.format(k))

        self.G_optimizer = Adam(G_optim_params, lr=self.opt_train['G_optimizer_lr'], weight_decay=0)
        self.D_optimizer = Adam(self.netD.parameters(), lr=self.opt_train['D_optimizer_lr'], weight_decay=0)

    # ----------------------------------------
    # define scheduler, only "MultiStepLR"
    # ----------------------------------------
    def define_scheduler(self):
        self.schedulers.append(lr_scheduler.MultiStepLR(self.G_optimizer,
                                                        self.opt_train['G_scheduler_milestones'],
                                                        self.opt_train['G_scheduler_gamma']
                                                        ))
        self.schedulers.append(lr_scheduler.MultiStepLR(self.D_optimizer,
                                                        self.opt_train['D_scheduler_milestones'],
                                                        self.opt_train['D_scheduler_gamma']
                                                        ))

    """
    # ----------------------------------------
    # Optimization during training with data
    # Testing/evaluation
    # ----------------------------------------
    """

    # ----------------------------------------
    # feed L/H data
    # ----------------------------------------
    def feed_data(self, data, need_H=True):
        self.L = data['L'].to(self.device)
        if need_H:
            self.H = data['H'].to(self.device)

    # ----------------------------------------
    # feed L to netG and get E
    # ----------------------------------------
    def netG_forward(self):
        self.E = self.netG(self.L)

    # ----------------------------------------
    # update parameters and get loss
    # ----------------------------------------
    def optimize_parameters(self, current_step):
        # ------------------------------------
        # optimize G
        # ------------------------------------
        for p in self.netD.parameters():
            p.requires_grad = False

        self.G_optimizer.zero_grad()
        self.netG_forward()
        loss_G_total = 0

        if current_step % self.D_update_ratio == 0 and current_step > self.D_init_iters:  # updata D first
            if self.opt_train['G_lossfn_weight'] > 0:
                G_loss = self.G_lossfn_weight * self.G_lossfn(self.E, self.H)
                loss_G_total += G_loss                 # 1) pixel loss
            if self.opt_train['F_lossfn_weight'] > 0:
                F_loss = self.F_lossfn_weight * self.F_lossfn(self.E, self.H)
                loss_G_total += F_loss                 # 2) VGG feature loss

            if self.opt['train']['gan_type'] in ['gan', 'lsgan', 'wgan', 'softplusgan']:
                pred_g_fake = self.netD(self.E)
                D_loss = self.D_lossfn_weight * self.D_lossfn(pred_g_fake, True)
            elif self.opt['train']['gan_type'] == 'ragan':
                pred_d_real = self.netD(self.H).detach()
                pred_g_fake = self.netD(self.E)
                D_loss = self.D_lossfn_weight * (
                        self.D_lossfn(pred_d_real - torch.mean(pred_g_fake, 0, True), False) +
                        self.D_lossfn(pred_g_fake - torch.mean(pred_d_real, 0, True), True)) / 2
            loss_G_total += D_loss                    # 3) GAN loss

            loss_G_total.backward()
            self.G_optimizer.step()

        # ------------------------------------
        # optimize D
        # ------------------------------------
        for p in self.netD.parameters():
            p.requires_grad = True

        self.D_optimizer.zero_grad()

        # In order to avoid the error in distributed training:
        # "Error detected in CudnnBatchNormBackward: RuntimeError: one of
        # the variables needed for gradient computation has been modified by
        # an inplace operation",
        # we separate the backwards for real and fake, and also detach the
        # tensor for calculating mean.
        if self.opt_train['gan_type'] in ['gan', 'lsgan', 'wgan', 'softplusgan']:
            # real
            pred_d_real = self.netD(self.H)                # 1) real data
            l_d_real = self.D_lossfn(pred_d_real, True)
            l_d_real.backward()
            # fake
            pred_d_fake = self.netD(self.E.detach().clone()) # 2) fake data, detach to avoid BP to G
            l_d_fake = self.D_lossfn(pred_d_fake, False)
            l_d_fake.backward()
        elif self.opt_train['gan_type'] == 'ragan':
            # real
            pred_d_fake = self.netD(self.E).detach()       # 1) fake data, detach to avoid BP to G
            pred_d_real = self.netD(self.H)                # 2) real data
            l_d_real = 0.5 * self.D_lossfn(pred_d_real - torch.mean(pred_d_fake, 0, True), True)
            l_d_real.backward()
            # fake
            pred_d_fake = self.netD(self.E.detach())
            l_d_fake = 0.5 * self.D_lossfn(pred_d_fake - torch.mean(pred_d_real.detach(), 0, True), False)
            l_d_fake.backward()

        self.D_optimizer.step()

        # ------------------------------------
        # record log
        # ------------------------------------
        if current_step % self.D_update_ratio == 0 and current_step > self.D_init_iters:
            if self.opt_train['G_lossfn_weight'] > 0:
                self.log_dict['G_loss'] = G_loss.item()
            if self.opt_train['F_lossfn_weight'] > 0:
                self.log_dict['F_loss'] = F_loss.item()
            self.log_dict['D_loss'] = D_loss.item()

        #self.log_dict['l_d_real'] = l_d_real.item()
        #self.log_dict['l_d_fake'] = l_d_fake.item()
        self.log_dict['D_real'] = torch.mean(pred_d_real.detach())
        self.log_dict['D_fake'] = torch.mean(pred_d_fake.detach())

        if self.opt_train['E_decay'] > 0:
            self.update_E(self.opt_train['E_decay'])

    # ----------------------------------------
    # test and inference
    # ----------------------------------------
    def test(self):
        self.netG.eval()
        with torch.no_grad():
            self.netG_forward()
        self.netG.train()

    # ----------------------------------------
    # get log_dict
    # ----------------------------------------
    def current_log(self):
        return self.log_dict

    # ----------------------------------------
    # get L, E, H images
    # ----------------------------------------
    def current_visuals(self, need_H=True):
        out_dict = OrderedDict()
        out_dict['L'] = self.L.detach()[0].float().cpu()
        out_dict['E'] = self.E.detach()[0].float().cpu()
        if need_H:
            out_dict['H'] = self.H.detach()[0].float().cpu()
        return out_dict

    """
    # ----------------------------------------
    # Information of netG, netD and netF
    # ----------------------------------------
    """

    # ----------------------------------------
    # print network
    # ----------------------------------------
    def print_network(self):
        msg = self.describe_network(self.netG)
        print(msg)
        if self.is_train:
            msg = self.describe_network(self.netD)
            print(msg)

    # ----------------------------------------
    # print params
    # ----------------------------------------
    def print_params(self):
        msg = self.describe_params(self.netG)
        print(msg)

    # ----------------------------------------
    # network information
    # ----------------------------------------
    def info_network(self):
        msg = self.describe_network(self.netG)
        if self.is_train:
            msg += self.describe_network(self.netD)
        return msg

    # ----------------------------------------
    # params information
    # ----------------------------------------
    def info_params(self):
        msg = self.describe_params(self.netG)
        return msg

