# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


import math
import os
import torch
from torch import nn as nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn import functional as F
from einops.layers.torch import Rearrange
from distutils.version import LooseVersion
from torch.utils.cpp_extension import load

module_path = os.path.dirname(__file__)
deform_attn_ext = load(
    'deform_attn',
    sources=[
        os.path.join(module_path, 'deform_attn_ext.cpp'),
        os.path.join(module_path, 'deform_attn_cuda_pt110.cpp' if LooseVersion(torch.__version__) >= LooseVersion(
            '1.10.0') else 'deform_attn_cuda_pt109.cpp'),
        os.path.join(module_path, 'deform_attn_cuda_kernel.cu'),
],
)


class Mlp(nn.Module):
    """ Multilayer perceptron.

    Args:
        x: (B, D, H, W, C)

    Returns:
        x: (B, D, H, W, C)
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class DeformAttnFunction(Function):

    @staticmethod
    def forward(ctx,
                q,
                kv,
                offset,
                kernel_h,
                kernel_w,
                stride=1,
                padding=0,
                dilation=1,
                attention_heads=1,
                deformable_groups=1,
                clip_size=1):
        ctx.kernel_h = kernel_h
        ctx.kernel_w = kernel_w
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.attention_heads = attention_heads
        ctx.deformable_groups = deformable_groups
        ctx.clip_size = clip_size
        if q.requires_grad or kv.requires_grad or offset.requires_grad:
            ctx.save_for_backward(q, kv, offset)
        output = q.new_empty(q.shape)
        ctx._bufs = [q.new_empty(0), q.new_empty(0), q.new_empty(0), q.new_empty(0), q.new_empty(0)]
        deform_attn_ext.deform_attn_forward(q, kv, offset, output,
                                                      ctx._bufs[0], ctx._bufs[1], ctx._bufs[2], ctx.kernel_h, ctx.kernel_w, ctx.stride,
                                                      ctx.stride, ctx.padding, ctx.padding, ctx.dilation, ctx.dilation,
                                                      ctx.attention_heads, ctx.deformable_groups, ctx.clip_size)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        if not grad_output.is_cuda:
            raise NotImplementedError
        q, kv, offset = ctx.saved_tensors
        grad_q = torch.zeros_like(q)
        grad_kv = torch.zeros_like(kv)
        grad_offset = torch.zeros_like(offset)
        deform_attn_ext.deform_attn_backward(q, kv, offset, ctx._bufs[0], ctx._bufs[1], ctx._bufs[2], ctx._bufs[3], ctx._bufs[4],
                                                       grad_q, grad_kv, grad_offset,
                                                       grad_output, ctx.kernel_h, ctx.kernel_w, ctx.stride,
                                                       ctx.stride, ctx.padding, ctx.padding, ctx.dilation, ctx.dilation,
                                                       ctx.attention_heads, ctx.deformable_groups, ctx.clip_size)

        return (grad_q, grad_kv, grad_offset, None, None, None, None, None, None, None, None)


deform_attn = DeformAttnFunction.apply


class DeformAttn(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 attention_window=[3, 3],
                 deformable_groups=12,
                 attention_heads=12,
                 clip_size=1):
        super(DeformAttn, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_h = attention_window[0]
        self.kernel_w = attention_window[1]
        self.attn_size = self.kernel_h * self.kernel_w
        self.deformable_groups = deformable_groups
        self.attention_heads = attention_heads
        self.clip_size = clip_size
        self.stride = 1
        self.padding = self.kernel_h//2
        self.dilation = 1

        self.proj_q = nn.Sequential(Rearrange('n d c h w -> n d h w c'),
                                    nn.Linear(self.in_channels, self.in_channels),
                                    Rearrange('n d h w c -> n d c h w'))
        self.proj_k = nn.Sequential(Rearrange('n d c h w -> n d h w c'),
                                    nn.Linear(self.in_channels, self.in_channels),
                                    Rearrange('n d h w c -> n d c h w'))
        self.proj_v = nn.Sequential(Rearrange('n d c h w -> n d h w c'),
                                    nn.Linear(self.in_channels, self.in_channels),
                                    Rearrange('n d h w c -> n d c h w'))
        self.mlp = nn.Sequential(Rearrange('n d c h w -> n d h w c'),
                                 Mlp(self.in_channels, self.in_channels * 2),
                                 Rearrange('n d h w c -> n d c h w'))

    def forward(self, q, k, v, offset):
        q = self.proj_q(q)
        kv = torch.cat([self.proj_k(k), self.proj_v(v)], 2)
        v = deform_attn(q, kv, offset, self.kernel_h, self.kernel_w, self.stride, self.padding, self.dilation,
                                     self.attention_heads, self.deformable_groups, self.clip_size)
        v =  v + self.mlp(v)
        return v


class DeformAttnPack(DeformAttn):
    """A Deformable Attention Encapsulation that acts as normal attention layers.

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        attention_window (int or tuple[int]): Attention window size. Default: [3, 3].
        attention_heads (int): Attention head number.  Default: 12.
        deformable_groups (int): Deformable offset groups.  Default: 12.
        clip_size (int): clip size. Default: 2.
    """

    def __init__(self, *args, **kwargs):
        super(DeformAttnPack, self).__init__(*args, **kwargs)

        self.conv_offset = nn.Conv2d(
            self.in_channels * (1 + self.clip_size),
            self.clip_size * self.deformable_groups * self.attn_size * 2,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            dilation=(1, 1),
            bias=True)
        self.init_weight()

    def init_weight(self):
        if hasattr(self, 'conv_offset'):
            self.conv_offset.weight.data.zero_()
            self.conv_offset.bias.data.zero_()

    def forward(self, q, k, v):
        out = self.conv_offset(torch.cat([q.flatten(1, 2), k.flatten(1, 2)], 1))
        o1, o2 = torch.chunk(out, 2, dim=1)
        offset = torch.cat((o1, o2), dim=1)

        q = self.proj_q(q)
        kv = torch.cat([self.proj_k(k), self.proj_v(v)], 2)
        v = deform_attn(q, kv, offset, self.kernel_h, self.kernel_w, self.stride, self.padding, self.dilation,
                                     self.attention_heads, self.deformable_groups, self.clip_size)
        v =  v + self.mlp(v)
        return v
