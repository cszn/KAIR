
import torch.nn as nn
import models.basicblock as B


"""
# --------------------------------------------
# References:
@article{2019Denoising,
  title={Denoising of MR images with Rician noise using a wider neural network and noise range division},
  author={ You, X.  and  Cao, N.  and  Lu, H.  and  Mao, M.  and  Wang, W. },
  journal={Magnetic Resonance Imaging},
  volume={64},
  year={2019},
}
# --------------------------------------------
"""


# --------------------------------------------
# WDNN
# --------------------------------------------
class WDNN(nn.Module):
    def __init__(self, in_nc=1, out_nc=1, nc=192, nb=8, act_mode='BR'):
        """
        # ------------------------------------
        in_nc: channel number of input
        out_nc: channel number of output
        nc: channel number
        nb: total number of conv layers
        act_mode: batch norm + activation function; 'BR' means BN+ReLU.
        # ------------------------------------
        Batch normalization and residual learning are
        beneficial to Gaussian denoising (especially
        for a single noise level).
        # ------------------------------------
        """
        super(WDNN, self).__init__()
        assert 'R' in act_mode or 'L' in act_mode, 'Examples of activation function: R, L, BR, BL, IR, IL'
        bias = True

        m_head = B.conv(in_nc, nc, mode='C'+act_mode[-1], bias=bias)
        m_body = [B.conv(nc, nc, mode='C'+act_mode, bias=bias) for _ in range(nb-2)]
        m_tail = B.conv(nc, out_nc, mode='C', bias=bias)

        self.model = B.sequential(m_head, *m_body, m_tail)

    def forward(self, x):
        n = self.model(x)
        return x-n