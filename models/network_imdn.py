import math
import torch.nn as nn
import models.basicblock as B


"""
# --------------------------------------------
# simplified information multi-distillation
# network (IMDN) for SR
# --------------------------------------------
References:
@inproceedings{hui2019lightweight,
  title={Lightweight Image Super-Resolution with Information Multi-distillation Network},
  author={Hui, Zheng and Gao, Xinbo and Yang, Yunchu and Wang, Xiumei},
  booktitle={Proceedings of the 27th ACM International Conference on Multimedia (ACM MM)},
  pages={2024--2032},
  year={2019}
}
@inproceedings{zhang2019aim,
  title={AIM 2019 Challenge on Constrained Super-Resolution: Methods and Results},
  author={Kai Zhang and Shuhang Gu and Radu Timofte and others},
  booktitle={IEEE International Conference on Computer Vision Workshops},
  year={2019}
}
# --------------------------------------------
"""


# --------------------------------------------
# modified version, https://github.com/Zheng222/IMDN
# first place solution for AIM 2019 challenge
# --------------------------------------------
class IMDN(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nc=64, nb=8, upscale=4, act_mode='L', upsample_mode='pixelshuffle', negative_slope=0.05):
        """
        in_nc: channel number of input
        out_nc: channel number of output
        nc: channel number
        nb: number of residual blocks
        upscale: up-scale factor
        act_mode: activation function
        upsample_mode: 'upconv' | 'pixelshuffle' | 'convtranspose'
        """
        super(IMDN, self).__init__()
        assert 'R' in act_mode or 'L' in act_mode, 'Examples of activation function: R, L, BR, BL, IR, IL'

        m_head = B.conv(in_nc, nc, mode='C')
        m_body = [B.IMDBlock(nc, nc, mode='C'+act_mode, negative_slope=negative_slope) for _ in range(nb)]
        m_body.append(B.conv(nc, nc, mode='C'))

        if upsample_mode == 'upconv':
            upsample_block = B.upsample_upconv
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.upsample_pixelshuffle
        elif upsample_mode == 'convtranspose':
            upsample_block = B.upsample_convtranspose
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))

        m_uper = upsample_block(nc, out_nc, mode=str(upscale))

        self.model = B.sequential(m_head, B.ShortcutBlock(B.sequential(*m_body)), *m_uper)

    def forward(self, x):
        x = self.model(x)
        return x
