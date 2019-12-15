import torch
import torch.nn as nn


"""
# --------------------------------------------
# Batch Normalization
# --------------------------------------------

# Kai Zhang (cskaizhang@gmail.com)
# https://github.com/cszn
# 01/Jan/2019
# --------------------------------------------
"""


# --------------------------------------------
# remove/delete specified layer
# --------------------------------------------
def deleteLayer(model, layer_type=nn.BatchNorm2d):
    ''' Kai Zhang, 11/Jan/2019.
    '''
    for k, m in list(model.named_children()):
        if isinstance(m, layer_type):
            del model._modules[k]
        deleteLayer(m, layer_type)


# --------------------------------------------
# merge bn, "conv+bn" --> "conv"
# --------------------------------------------
def merge_bn(model):
    ''' Kai Zhang, 11/Jan/2019.
    merge all 'Conv+BN' (or 'TConv+BN') into 'Conv' (or 'TConv')
    based on https://github.com/pytorch/pytorch/pull/901
    '''
    prev_m = None
    for k, m in list(model.named_children()):
        if (isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d)) and (isinstance(prev_m, nn.Conv2d) or isinstance(prev_m, nn.Linear) or isinstance(prev_m, nn.ConvTranspose2d)):

            w = prev_m.weight.data

            if prev_m.bias is None:
                zeros = torch.Tensor(prev_m.out_channels).zero_().type(w.type())
                prev_m.bias = nn.Parameter(zeros)
            b = prev_m.bias.data

            invstd = m.running_var.clone().add_(m.eps).pow_(-0.5)
            if isinstance(prev_m, nn.ConvTranspose2d):
                w.mul_(invstd.view(1, w.size(1), 1, 1).expand_as(w))
            else:
                w.mul_(invstd.view(w.size(0), 1, 1, 1).expand_as(w))
            b.add_(-m.running_mean).mul_(invstd)
            if m.affine:
                if isinstance(prev_m, nn.ConvTranspose2d):
                    w.mul_(m.weight.data.view(1, w.size(1), 1, 1).expand_as(w))
                else:
                    w.mul_(m.weight.data.view(w.size(0), 1, 1, 1).expand_as(w))
                b.mul_(m.weight.data).add_(m.bias.data)

            del model._modules[k]
        prev_m = m
        merge_bn(m)


# --------------------------------------------
# add bn, "conv" --> "conv+bn"
# --------------------------------------------
def add_bn(model):
    ''' Kai Zhang, 11/Jan/2019.
    '''
    for k, m in list(model.named_children()):
        if (isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose2d)):
            b = nn.BatchNorm2d(m.out_channels, momentum=0.1, affine=True)
            b.weight.data.fill_(1)
            new_m = nn.Sequential(model._modules[k], b)
            model._modules[k] = new_m
        add_bn(m)


# --------------------------------------------
# tidy model after removing bn
# --------------------------------------------
def tidy_sequential(model):
    ''' Kai Zhang, 11/Jan/2019.
    '''
    for k, m in list(model.named_children()):
        if isinstance(m, nn.Sequential):
            if m.__len__() == 1:
                model._modules[k] = m.__getitem__(0)
        tidy_sequential(m)
