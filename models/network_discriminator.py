import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm
import models.basicblock as B
import functools
import numpy as np


"""
# --------------------------------------------
# Discriminator_PatchGAN
# Discriminator_UNet
# --------------------------------------------
"""


# --------------------------------------------
# PatchGAN discriminator
# If n_layers = 3, then the receptive field is 70x70
# --------------------------------------------
class Discriminator_PatchGAN(nn.Module):
    def __init__(self, input_nc=3, ndf=64, n_layers=3, norm_type='spectral'):
        '''PatchGAN discriminator, receptive field = 70x70 if n_layers = 3
        Args:
            input_nc: number of input channels 
            ndf: base channel number
            n_layers: number of conv layer with stride 2
            norm_type:  'batch', 'instance', 'spectral', 'batchspectral', instancespectral'
        Returns:
            tensor: score
        '''
        super(Discriminator_PatchGAN, self).__init__()
        self.n_layers = n_layers
        norm_layer = self.get_norm_layer(norm_type=norm_type)

        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        sequence = [[self.use_spectral_norm(nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), norm_type), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[self.use_spectral_norm(nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw), norm_type),
                          norm_layer(nf), 
                          nn.LeakyReLU(0.2, True)]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[self.use_spectral_norm(nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw), norm_type),
                      norm_layer(nf),
                      nn.LeakyReLU(0.2, True)]]

        sequence += [[self.use_spectral_norm(nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw), norm_type)]]

        self.model = nn.Sequential()
        for n in range(len(sequence)):
            self.model.add_module('child' + str(n), nn.Sequential(*sequence[n]))

        self.model.apply(self.weights_init)

    def use_spectral_norm(self, module, norm_type='spectral'):
        if 'spectral' in norm_type:
            return spectral_norm(module)
        return module

    def get_norm_layer(self, norm_type='instance'):
        if 'batch' in norm_type:
            norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
        elif 'instance' in norm_type:
            norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
        else:
            norm_layer = functools.partial(nn.Identity)
        return norm_layer

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm2d') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    def forward(self, x):
        return self.model(x)


class Discriminator_UNet(nn.Module):
    """Defines a U-Net discriminator with spectral normalization (SN)"""

    def __init__(self, input_nc=3, ndf=64):
        super(Discriminator_UNet, self).__init__()
        norm = spectral_norm

        self.conv0 = nn.Conv2d(input_nc, ndf, kernel_size=3, stride=1, padding=1)

        self.conv1 = norm(nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False))
        self.conv2 = norm(nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False))
        self.conv3 = norm(nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False))
        # upsample
        self.conv4 = norm(nn.Conv2d(ndf * 8, ndf * 4, 3, 1, 1, bias=False))
        self.conv5 = norm(nn.Conv2d(ndf * 4, ndf * 2, 3, 1, 1, bias=False))
        self.conv6 = norm(nn.Conv2d(ndf * 2, ndf, 3, 1, 1, bias=False))

        # extra
        self.conv7 = norm(nn.Conv2d(ndf, ndf, 3, 1, 1, bias=False))
        self.conv8 = norm(nn.Conv2d(ndf, ndf, 3, 1, 1, bias=False))

        self.conv9 = nn.Conv2d(ndf, 1, 3, 1, 1)
        print('using the UNet discriminator')

    def forward(self, x):
        x0 = F.leaky_relu(self.conv0(x), negative_slope=0.2, inplace=True)
        x1 = F.leaky_relu(self.conv1(x0), negative_slope=0.2, inplace=True)
        x2 = F.leaky_relu(self.conv2(x1), negative_slope=0.2, inplace=True)
        x3 = F.leaky_relu(self.conv3(x2), negative_slope=0.2, inplace=True)

        # upsample
        x3 = F.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=False)
        x4 = F.leaky_relu(self.conv4(x3), negative_slope=0.2, inplace=True)

        x4 = x4 + x2
        x4 = F.interpolate(x4, scale_factor=2, mode='bilinear', align_corners=False)
        x5 = F.leaky_relu(self.conv5(x4), negative_slope=0.2, inplace=True)

        x5 = x5 + x1
        x5 = F.interpolate(x5, scale_factor=2, mode='bilinear', align_corners=False)
        x6 = F.leaky_relu(self.conv6(x5), negative_slope=0.2, inplace=True)

        x6 = x6 + x0

        # extra
        out = F.leaky_relu(self.conv7(x6), negative_slope=0.2, inplace=True)
        out = F.leaky_relu(self.conv8(out), negative_slope=0.2, inplace=True)
        out = self.conv9(out)

        return out


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
