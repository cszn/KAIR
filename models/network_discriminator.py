import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
import models.basicblock as B
import functools
import numpy as np


"""
# --------------------------------------------
# Discriminator_VGG_96
# Discriminator_VGG_128
# Discriminator_VGG_192
# Discriminator_VGG_128_SN
# --------------------------------------------
"""


# --------------------------------------------
# PatchGAN discriminator
# If n_layers = 3, then the receptive field is 70x70
# --------------------------------------------
class Discriminator_PatchGAN(nn.Module):
    def __init__(self, input_nc=3, ndf=64, n_layers=3, norm_type='batch'):
        super(Discriminator_PatchGAN, self).__init__()
        self.n_layers = n_layers

        '''
        'batch'
        'instance'
        'spectral'
        'batchspectral'
        'none'
        '''
        if norm_type == 'batch':
            norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
            use_sp_norm = False
            use_bias = False
        elif norm_type == 'instance':
            norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
            use_sp_norm = False
            use_bias = True
        elif norm_type == 'spectral':
            norm_layer = None
            use_sp_norm = True
            use_bias = True
        elif norm_type == 'batchspectral':
            norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
            use_sp_norm = True
            use_bias = False
        elif norm_type == 'none':
            norm_layer = None
            use_sp_norm = False
            use_bias = True
        else:
            raise NotImplementedError('normalization layer [%s] is not found' % norm_type)

        kw = 4
        padw = int(np.ceil((kw-1.0)/2))
        sequence = [
            self.use_spectral_norm(nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), use_sp_norm),
            nn.LeakyReLU(0.2)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                self.use_spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias), use_sp_norm),
                norm_layer(ndf * nf_mult) if norm_layer else None,
                nn.LeakyReLU(0.2)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            self.use_spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias), use_sp_norm),
            norm_layer(ndf * nf_mult) if norm_layer else None,
            nn.LeakyReLU(0.2)
        ]
        sequence += [self.use_spectral_norm(nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw), use_sp_norm)]
        
        sequence_new = []
        for n in range(len(sequence)):
            if sequence[n] is not None:
                sequence_new.append(sequence[n])

        self.model = nn.Sequential(*sequence_new)

    def use_spectral_norm(self, module, mode=False):
        if mode:
            return spectral_norm(module)
        return module

    def forward(self, input):
        return self.model(input)


# --------------------------------------------
# VGG style Discriminator with 96x96 input
# --------------------------------------------
class Discriminator_VGG_96(nn.Module):
    def __init__(self, in_nc=3, base_nc=64, ac_type='BL'):
        super(Discriminator_VGG_96, self).__init__()
        # features
        # hxw, c
        # 96, 64
        conv0 = B.conv(in_nc, base_nc, kernel_size=3, mode='C')
        conv1 = B.conv(base_nc, base_nc, kernel_size=4, stride=2, mode='C'+ac_type)
        # 48, 64
        conv2 = B.conv(base_nc, base_nc*2, kernel_size=3, stride=1, mode='C'+ac_type)
        conv3 = B.conv(base_nc*2, base_nc*2, kernel_size=4, stride=2, mode='C'+ac_type)
        # 24, 128
        conv4 = B.conv(base_nc*2, base_nc*4, kernel_size=3, stride=1, mode='C'+ac_type)
        conv5 = B.conv(base_nc*4, base_nc*4, kernel_size=4, stride=2, mode='C'+ac_type)
        # 12, 256
        conv6 = B.conv(base_nc*4, base_nc*8, kernel_size=3, stride=1, mode='C'+ac_type)
        conv7 = B.conv(base_nc*8, base_nc*8, kernel_size=4, stride=2, mode='C'+ac_type)
        # 6, 512
        conv8 = B.conv(base_nc*8, base_nc*8, kernel_size=3, stride=1, mode='C'+ac_type)
        conv9 = B.conv(base_nc*8, base_nc*8, kernel_size=4, stride=2, mode='C'+ac_type)
        # 3, 512
        self.features = B.sequential(conv0, conv1, conv2, conv3, conv4,
                                     conv5, conv6, conv7, conv8, conv9)

        # classifier
        self.classifier = nn.Sequential(
            nn.Linear(512 * 3 * 3, 100), nn.LeakyReLU(0.2, True), nn.Linear(100, 1))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# --------------------------------------------
# VGG style Discriminator with 128x128 input
# --------------------------------------------
class Discriminator_VGG_128(nn.Module):
    def __init__(self, in_nc=3, base_nc=64, ac_type='BL'):
        super(Discriminator_VGG_128, self).__init__()
        # features
        # hxw, c
        # 128, 64
        conv0 = B.conv(in_nc, base_nc, kernel_size=3, mode='C')
        conv1 = B.conv(base_nc, base_nc, kernel_size=4, stride=2, mode='C'+ac_type)
        # 64, 64
        conv2 = B.conv(base_nc, base_nc*2, kernel_size=3, stride=1, mode='C'+ac_type)
        conv3 = B.conv(base_nc*2, base_nc*2, kernel_size=4, stride=2, mode='C'+ac_type)
        # 32, 128
        conv4 = B.conv(base_nc*2, base_nc*4, kernel_size=3, stride=1, mode='C'+ac_type)
        conv5 = B.conv(base_nc*4, base_nc*4, kernel_size=4, stride=2, mode='C'+ac_type)
        # 16, 256
        conv6 = B.conv(base_nc*4, base_nc*8, kernel_size=3, stride=1, mode='C'+ac_type)
        conv7 = B.conv(base_nc*8, base_nc*8, kernel_size=4, stride=2, mode='C'+ac_type)
        # 8, 512
        conv8 = B.conv(base_nc*8, base_nc*8, kernel_size=3, stride=1, mode='C'+ac_type)
        conv9 = B.conv(base_nc*8, base_nc*8, kernel_size=4, stride=2, mode='C'+ac_type)
        # 4, 512
        self.features = B.sequential(conv0, conv1, conv2, conv3, conv4,
                                     conv5, conv6, conv7, conv8, conv9)

        # classifier
        self.classifier = nn.Sequential(nn.Linear(512 * 4 * 4, 100), 
                                        nn.LeakyReLU(0.2, True), 
                                        nn.Linear(100, 1))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# --------------------------------------------
# VGG style Discriminator with 192x192 input
# --------------------------------------------
class Discriminator_VGG_192(nn.Module):
    def __init__(self, in_nc=3, base_nc=64, ac_type='BL'):
        super(Discriminator_VGG_192, self).__init__()
        # features
        # hxw, c
        # 192, 64
        conv0 = B.conv(in_nc, base_nc, kernel_size=3, mode='C')
        conv1 = B.conv(base_nc, base_nc, kernel_size=4, stride=2, mode='C'+ac_type)
        # 96, 64
        conv2 = B.conv(base_nc, base_nc*2, kernel_size=3, stride=1, mode='C'+ac_type)
        conv3 = B.conv(base_nc*2, base_nc*2, kernel_size=4, stride=2, mode='C'+ac_type)
        # 48, 128
        conv4 = B.conv(base_nc*2, base_nc*4, kernel_size=3, stride=1, mode='C'+ac_type)
        conv5 = B.conv(base_nc*4, base_nc*4, kernel_size=4, stride=2, mode='C'+ac_type)
        # 24, 256
        conv6 = B.conv(base_nc*4, base_nc*8, kernel_size=3, stride=1, mode='C'+ac_type)
        conv7 = B.conv(base_nc*8, base_nc*8, kernel_size=4, stride=2, mode='C'+ac_type)
        # 12, 512
        conv8 = B.conv(base_nc*8, base_nc*8, kernel_size=3, stride=1, mode='C'+ac_type)
        conv9 = B.conv(base_nc*8, base_nc*8, kernel_size=4, stride=2, mode='C'+ac_type)
        # 6, 512
        conv10 = B.conv(base_nc*8, base_nc*8, kernel_size=3, stride=1, mode='C'+ac_type)
        conv11 = B.conv(base_nc*8, base_nc*8, kernel_size=4, stride=2, mode='C'+ac_type)
        # 3, 512
        self.features = B.sequential(conv0, conv1, conv2, conv3, conv4, conv5,
                                     conv6, conv7, conv8, conv9, conv10, conv11)

        # classifier
        self.classifier = nn.Sequential(nn.Linear(512 * 3 * 3, 100),
                                        nn.LeakyReLU(0.2, True),
                                        nn.Linear(100, 1))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# --------------------------------------------
# SN-VGG style Discriminator with 128x128 input
# --------------------------------------------
class Discriminator_VGG_128_SN(nn.Module):
    def __init__(self):
        super(Discriminator_VGG_128_SN, self).__init__()
        # features
        # hxw, c
        # 128, 64
        self.lrelu = nn.LeakyReLU(0.2, True)

        self.conv0 = spectral_norm(nn.Conv2d(3, 64, 3, 1, 1))
        self.conv1 = spectral_norm(nn.Conv2d(64, 64, 4, 2, 1))
        # 64, 64
        self.conv2 = spectral_norm(nn.Conv2d(64, 128, 3, 1, 1))
        self.conv3 = spectral_norm(nn.Conv2d(128, 128, 4, 2, 1))
        # 32, 128
        self.conv4 = spectral_norm(nn.Conv2d(128, 256, 3, 1, 1))
        self.conv5 = spectral_norm(nn.Conv2d(256, 256, 4, 2, 1))
        # 16, 256
        self.conv6 = spectral_norm(nn.Conv2d(256, 512, 3, 1, 1))
        self.conv7 = spectral_norm(nn.Conv2d(512, 512, 4, 2, 1))
        # 8, 512
        self.conv8 = spectral_norm(nn.Conv2d(512, 512, 3, 1, 1))
        self.conv9 = spectral_norm(nn.Conv2d(512, 512, 4, 2, 1))
        # 4, 512

        # classifier
        self.linear0 = spectral_norm(nn.Linear(512 * 4 * 4, 100))
        self.linear1 = spectral_norm(nn.Linear(100, 1))

    def forward(self, x):
        x = self.lrelu(self.conv0(x))
        x = self.lrelu(self.conv1(x))
        x = self.lrelu(self.conv2(x))
        x = self.lrelu(self.conv3(x))
        x = self.lrelu(self.conv4(x))
        x = self.lrelu(self.conv5(x))
        x = self.lrelu(self.conv6(x))
        x = self.lrelu(self.conv7(x))
        x = self.lrelu(self.conv8(x))
        x = self.lrelu(self.conv9(x))
        x = x.view(x.size(0), -1)
        x = self.lrelu(self.linear0(x))
        x = self.linear1(x)
        return x


if __name__ == '__main__':

    x = torch.rand(1, 3, 96, 96)
    net = Discriminator_VGG_96()
    net.eval()
    with torch.no_grad():
        y = net(x)
    print(y.size())

    x = torch.rand(1, 3, 128, 128)
    net = Discriminator_VGG_128()
    net.eval()
    with torch.no_grad():
        y = net(x)
    print(y.size())

    x = torch.rand(1, 3, 192, 192)
    net = Discriminator_VGG_192()
    net.eval()
    with torch.no_grad():
        y = net(x)
    print(y.size())

    x = torch.rand(1, 3, 128, 128)
    net = Discriminator_VGG_128_SN()
    net.eval()
    with torch.no_grad():
        y = net(x)
    print(y.size())

    # run models/network_discriminator.py
