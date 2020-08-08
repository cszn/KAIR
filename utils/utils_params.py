import torch

import torchvision

from models import basicblock as B

def show_kv(net):
    for k, v in net.items():
        print(k)

# should run train debug mode first to get an initial model
#crt_net = torch.load('../../experiments/debug_SRResNet_bicx4_in3nf64nb16/models/8_G.pth')
#
#for k, v in crt_net.items():
#    print(k)
#for k, v in crt_net.items():
#    if k in pretrained_net:
#        crt_net[k] = pretrained_net[k]
#        print('replace ... ', k)

# x2 -> x4
#crt_net['model.5.weight'] = pretrained_net['model.2.weight']
#crt_net['model.5.bias'] = pretrained_net['model.2.bias']
#crt_net['model.8.weight'] = pretrained_net['model.5.weight']
#crt_net['model.8.bias'] = pretrained_net['model.5.bias']
#crt_net['model.10.weight'] = pretrained_net['model.7.weight']
#crt_net['model.10.bias'] = pretrained_net['model.7.bias']
#torch.save(crt_net, '../pretrained_tmp.pth')

# x2 -> x3
'''
in_filter = pretrained_net['model.2.weight'] # 256, 64, 3, 3
new_filter = torch.Tensor(576, 64, 3, 3)
new_filter[0:256, :, :, :] = in_filter
new_filter[256:512, :, :, :] = in_filter
new_filter[512:, :, :, :] = in_filter[0:576-512, :, :, :]
crt_net['model.2.weight'] = new_filter

in_bias = pretrained_net['model.2.bias']  # 256, 64, 3, 3
new_bias = torch.Tensor(576)
new_bias[0:256] = in_bias
new_bias[256:512] = in_bias
new_bias[512:] = in_bias[0:576 - 512]
crt_net['model.2.bias'] = new_bias

torch.save(crt_net, '../pretrained_tmp.pth')
'''

# x2 -> x8
'''
crt_net['model.5.weight'] = pretrained_net['model.2.weight']
crt_net['model.5.bias'] = pretrained_net['model.2.bias']
crt_net['model.8.weight'] = pretrained_net['model.2.weight']
crt_net['model.8.bias'] = pretrained_net['model.2.bias']
crt_net['model.11.weight'] = pretrained_net['model.5.weight']
crt_net['model.11.bias'] = pretrained_net['model.5.bias']
crt_net['model.13.weight'] = pretrained_net['model.7.weight']
crt_net['model.13.bias'] = pretrained_net['model.7.bias']
torch.save(crt_net, '../pretrained_tmp.pth')
'''

# x3/4/8 RGB -> Y

def rgb2gray_net(net, only_input=True):

    if only_input:
        in_filter = net['0.weight']
        in_new_filter = in_filter[:,0,:,:]*0.2989 + in_filter[:,1,:,:]*0.587 + in_filter[:,2,:,:]*0.114
        in_new_filter.unsqueeze_(1)
        net['0.weight'] = in_new_filter

#    out_filter = pretrained_net['model.13.weight']
#    out_new_filter = out_filter[0, :, :, :] * 0.2989 + out_filter[1, :, :, :] * 0.587 + \
#        out_filter[2, :, :, :] * 0.114
#    out_new_filter.unsqueeze_(0)
#    crt_net['model.13.weight'] = out_new_filter
#    out_bias = pretrained_net['model.13.bias']
#    out_new_bias = out_bias[0] * 0.2989 + out_bias[1] * 0.587 + out_bias[2] * 0.114
#    out_new_bias = torch.Tensor(1).fill_(out_new_bias)
#    crt_net['model.13.bias'] = out_new_bias

#    torch.save(crt_net, '../pretrained_tmp.pth')

    return net



if __name__ == '__main__':
    
    net = torchvision.models.vgg19(pretrained=True)
    for k,v in net.features.named_parameters():
        if k=='0.weight':
            in_new_filter = v[:,0,:,:]*0.2989 + v[:,1,:,:]*0.587 + v[:,2,:,:]*0.114
            in_new_filter.unsqueeze_(1)
            v = in_new_filter
            print(v.shape)
            print(v[0,0,0,0])
        if k=='0.bias':
            in_new_bias = v
            print(v[0])

    print(net.features[0])

    net.features[0] = B.conv(1, 64, mode='C') 

    print(net.features[0])
    net.features[0].weight.data=in_new_filter
    net.features[0].bias.data=in_new_bias

    for k,v in net.features.named_parameters():
        if k=='0.weight':
            print(v[0,0,0,0])
        if k=='0.bias':
            print(v[0])

    # transfer parameters of old model to new one
    model_old = torch.load(model_path)
    state_dict = model.state_dict()
    for ((key, param),(key2, param2)) in zip(model_old.items(), state_dict.items()):
        state_dict[key2] = param
        print([key, key2])
       # print([param.size(), param2.size()])
    torch.save(state_dict, 'model_new.pth') 


   # rgb2gray_net(net)









