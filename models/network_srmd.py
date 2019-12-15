
import torch.nn as nn
import models.basicblock as B
import torch

"""
# --------------------------------------------
# SRMD (15 conv layers)
# --------------------------------------------
Reference:
@inproceedings{zhang2018learning,
  title={Learning a single convolutional super-resolution network for multiple degradations},
  author={Zhang, Kai and Zuo, Wangmeng and Zhang, Lei},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition},
  pages={3262--3271},
  year={2018}
}
http://openaccess.thecvf.com/content_cvpr_2018/papers/Zhang_Learning_a_Single_CVPR_2018_paper.pdf
"""


# --------------------------------------------
# SRMD   (SRMD,   in_nc = 3+15+1 = 19)
# SRMD   (SRMDNF, in_nc = 3+15   = 18)
# --------------------------------------------
class SRMD(nn.Module):
    def __init__(self, in_nc=19, out_nc=3, nc=128, nb=12, upscale=4, act_mode='R', upsample_mode='pixelshuffle'):
        """
        # ------------------------------------
        in_nc: channel number of input, default: 3+15
        out_nc: channel number of output
        nc: channel number
        nb: total number of conv layers
        upscale: scale factor
        act_mode: batch norm + activation function; 'BR' means BN+ReLU
        upsample_mode: default 'pixelshuffle' = conv + pixelshuffle
        # ------------------------------------
        """
        super(SRMD, self).__init__()
        assert 'R' in act_mode or 'L' in act_mode, 'Examples of activation function: R, L, BR, BL, IR, IL'
        bias = True

        if upsample_mode == 'upconv':
            upsample_block = B.upsample_upconv
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.upsample_pixelshuffle
        elif upsample_mode == 'convtranspose':
            upsample_block = B.upsample_convtranspose
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))

        m_head = B.conv(in_nc, nc, mode='C'+act_mode[-1], bias=bias)
        m_body = [B.conv(nc, nc, mode='C'+act_mode, bias=bias) for _ in range(nb-2)]
        m_tail = upsample_block(nc, out_nc, mode=str(upscale), bias=bias)

        self.model = B.sequential(m_head, *m_body, m_tail)

#    def forward(self, x, k_pca):
#        m = k_pca.repeat(1, 1, x.size()[-2], x.size()[-1])
#        x = torch.cat((x, m), 1)
#        x = self.body(x)

    def forward(self, x):

        x = self.model(x)

        return x


if __name__ == '__main__':
    from utils import utils_model
    model = SRMD(in_nc=18, out_nc=3, nc=64, nb=15, upscale=4, act_mode='R', upsample_mode='pixelshuffle')
    print(utils_model.describe_model(model))

    x = torch.randn((2, 3, 100, 100))
    k_pca = torch.randn(2, 15, 1, 1)
    x = model(x, k_pca)
    print(x.shape)

    #  run models/network_srmd.py

