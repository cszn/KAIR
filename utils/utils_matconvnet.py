# -*- coding: utf-8 -*-
import numpy as np
import torch
from collections import OrderedDict

# import scipy.io as io
import hdf5storage

"""
# --------------------------------------------
# Convert matconvnet SimpleNN model into pytorch model
# --------------------------------------------
# Kai Zhang (cskaizhang@gmail.com)
# https://github.com/cszn
# 28/Nov/2019
# --------------------------------------------
"""


def weights2tensor(x, squeeze=False, in_features=None, out_features=None):
    """Modified version of https://github.com/albanie/pytorch-mcn
    Adjust memory layout and load weights as torch tensor
    Args:
        x (ndaray): a numpy array, corresponding to a set of network weights
           stored in column major order
        squeeze (bool) [False]: whether to squeeze the tensor (i.e. remove
           singletons from the trailing dimensions. So after converting to
           pytorch layout (C_out, C_in, H, W), if the shape is (A, B, 1, 1)
           it will be reshaped to a matrix with shape (A,B).
        in_features (int :: None): used to reshape weights for a linear block.
        out_features (int :: None): used to reshape weights for a linear block.
    Returns:
        torch.tensor: a permuted sets of weights, matching the pytorch layout
        convention
    """
    if x.ndim == 4:
        x = x.transpose((3, 2, 0, 1))
# for FFDNet, pixel-shuffle layer
#        if x.shape[1]==13:
#            x=x[:,[0,2,1,3,  4,6,5,7, 8,10,9,11, 12],:,:]
#        if x.shape[0]==12:   
#            x=x[[0,2,1,3,  4,6,5,7, 8,10,9,11],:,:,:]
#        if x.shape[1]==5:
#            x=x[:,[0,2,1,3,  4],:,:]
#        if x.shape[0]==4:   
#            x=x[[0,2,1,3],:,:,:]
## for SRMD, pixel-shuffle layer
#        if x.shape[0]==12:   
#            x=x[[0,2,1,3,  4,6,5,7, 8,10,9,11],:,:,:]
#        if x.shape[0]==27:
#            x=x[[0,3,6,1,4,7,2,5,8, 0+9,3+9,6+9,1+9,4+9,7+9,2+9,5+9,8+9, 0+18,3+18,6+18,1+18,4+18,7+18,2+18,5+18,8+18],:,:,:]
#        if x.shape[0]==48:   
#            x=x[[0,4,8,12,1,5,9,13,2,6,10,14,3,7,11,15,  0+16,4+16,8+16,12+16,1+16,5+16,9+16,13+16,2+16,6+16,10+16,14+16,3+16,7+16,11+16,15+16,  0+32,4+32,8+32,12+32,1+32,5+32,9+32,13+32,2+32,6+32,10+32,14+32,3+32,7+32,11+32,15+32],:,:,:]

    elif x.ndim == 3:  # add by Kai
        x = x[:,:,:,None]
        x = x.transpose((3, 2, 0, 1))
    elif x.ndim == 2:
        if x.shape[1] == 1:
            x = x.flatten()
    if squeeze:
        if in_features and out_features:
            x = x.reshape((out_features, in_features))
        x = np.squeeze(x)
    return torch.from_numpy(np.ascontiguousarray(x))


def save_model(network, save_path):
    state_dict = network.state_dict()
    for key, param in state_dict.items():
        state_dict[key] = param.cpu()
    torch.save(state_dict, save_path)


if __name__ == '__main__':
    
    
#    from utils import utils_logger
#    import logging
#    utils_logger.logger_info('a', 'a.log')
#    logger = logging.getLogger('a')
#    
    # mcn = hdf5storage.loadmat('/model_zoo/matfile/FFDNet_Clip_gray.mat')
    mcn = hdf5storage.loadmat('models/modelcolor.mat')
    
    
    #logger.info(mcn['CNNdenoiser'][0][0][0][1][0][0][0][0])

    mat_net = OrderedDict()
    for idx in range(25):
        mat_net[str(idx)] = OrderedDict()
        count = -1
        
        print(idx)
        for i in range(13):
            
            if mcn['CNNdenoiser'][0][idx][0][i][0][0][0][0] == 'conv':
                
                count += 1
                w = mcn['CNNdenoiser'][0][idx][0][i][0][1][0][0]
               # print(w.shape)
                w = weights2tensor(w)
               # print(w.shape)
                
                b = mcn['CNNdenoiser'][0][idx][0][i][0][1][0][1]
                b = weights2tensor(b)
                print(b.shape)

                mat_net[str(idx)]['model.{:d}.weight'.format(count*2)] = w
                mat_net[str(idx)]['model.{:d}.bias'.format(count*2)] = b

    torch.save(mat_net, 'model_zoo/modelcolor.pth')
   


#    from models.network_dncnn import IRCNN as net
#    network = net(in_nc=3, out_nc=3, nc=64)
#    state_dict = network.state_dict()
#
#    #show_kv(state_dict)
#
#    for i in range(len(mcn['net'][0][0][0])):
#        print(mcn['net'][0][0][0][i][0][0][0][0])
#
#    count = -1
#    mat_net = OrderedDict()
#    for i in range(len(mcn['net'][0][0][0])):
#        if mcn['net'][0][0][0][i][0][0][0][0] == 'conv':
#            
#            count += 1
#            w = mcn['net'][0][0][0][i][0][1][0][0]
#            print(w.shape)
#            w = weights2tensor(w)
#            print(w.shape)
#            
#            b = mcn['net'][0][0][0][i][0][1][0][1]
#            b = weights2tensor(b)
#            print(b.shape)
#            
#            mat_net['model.{:d}.weight'.format(count*2)] = w
#            mat_net['model.{:d}.bias'.format(count*2)] = b
#
#    torch.save(mat_net, 'E:/pytorch/KAIR_ongoing/model_zoo/ffdnet_gray_clip.pth')
#    
#    
#
#    crt_net = torch.load('E:/pytorch/KAIR_ongoing/model_zoo/imdn_x4.pth')
#    def show_kv(net):
#        for k, v in net.items():
#            print(k)
#
#    show_kv(crt_net)


#    from models.network_dncnn import DnCNN as net
#    network = net(in_nc=2, out_nc=1, nc=64, nb=20, act_mode='R')

#    from models.network_srmd import SRMD as net
#    #network = net(in_nc=1, out_nc=1, nc=64, nb=15, act_mode='R')
#    network = net(in_nc=19, out_nc=3, nc=128, nb=12, upscale=4, act_mode='R', upsample_mode='pixelshuffle')
#    
#    from models.network_rrdb import RRDB as net
#    network = net(in_nc=3, out_nc=3, nc=64, nb=23, gc=32, upscale=4, act_mode='L', upsample_mode='upconv')
#    
#    state_dict = network.state_dict()
#    for key, param in state_dict.items():
#        print(key)
#    from models.network_imdn import IMDN as net
#    network = net(in_nc=3, out_nc=3, nc=64, nb=8, upscale=4, act_mode='L', upsample_mode='pixelshuffle')
#    state_dict = network.state_dict()
#    mat_net = OrderedDict()
#    for ((key, param),(key2, param2)) in zip(state_dict.items(), crt_net.items()):
#        mat_net[key] = param2
#    torch.save(mat_net, 'model_zoo/imdn_x4_1.pth') 
#        

#    net_old = torch.load('net_old.pth')
#    def show_kv(net):
#        for k, v in net.items():
#            print(k)
#
#    show_kv(net_old)
#    from models.network_dpsr import MSRResNet_prior as net
#    model = net(in_nc=4, out_nc=3, nc=96, nb=16, upscale=4, act_mode='R', upsample_mode='pixelshuffle')
#    state_dict = network.state_dict()
#    net_new = OrderedDict()
#    for ((key, param),(key_old, param_old)) in zip(state_dict.items(), net_old.items()):
#        net_new[key] = param_old
#    torch.save(net_new, 'net_new.pth') 


   # print(key)
      #  print(param.size())



    # run utils/utils_matconvnet.py
