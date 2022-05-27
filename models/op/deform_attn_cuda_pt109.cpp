// modify from
// https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/blob/mmdetection/mmdet/ops/dcn/src/deform_conv_cuda.c

#include <torch/extension.h>
#include <ATen/DeviceGuard.h>
#include <iostream>

#include <cmath>
#include <vector>

void deformable_im2col(const at::Tensor data_im, const at::Tensor data_offset,
                       const int channels, const int height, const int width,
                       const int ksize_h, const int ksize_w, const int pad_h,
                       const int pad_w, const int stride_h, const int stride_w,
                       const int dilation_h, const int dilation_w,
                       const int parallel_imgs, const int deform_group,
                       at::Tensor data_col);

void deformable_col2im(const at::Tensor data_col, const at::Tensor data_offset,
                       const int channels, const int height, const int width,
                       const int ksize_h, const int ksize_w, const int pad_h,
                       const int pad_w, const int stride_h, const int stride_w,
                       const int dilation_h, const int dilation_w,
                       const int parallel_imgs, const int deform_group,
                       at::Tensor grad_im);

void deformable_col2im_coord(
    const at::Tensor data_col, const at::Tensor data_im,
    const at::Tensor data_offset, const int channels, const int height,
    const int width, const int ksize_h, const int ksize_w, const int pad_h,
    const int pad_w, const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w, const int parallel_imgs,
    const int deform_group, at::Tensor grad_offset);

void modulated_deformable_im2col_cuda(
    const at::Tensor data_im, const at::Tensor data_offset,
    const at::Tensor data_mask, const int batch_size, const int channels,
    const int height_im, const int width_im, const int height_col,
    const int width_col, const int kernel_h, const int kenerl_w,
    const int pad_h, const int pad_w, const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w, const int deform_group,
    at::Tensor data_col);

void modulated_deformable_col2im_cuda(
    const at::Tensor data_col, const at::Tensor data_offset,
    const at::Tensor data_mask, const int batch_size, const int channels,
    const int height_im, const int width_im, const int height_col,
    const int width_col, const int kernel_h, const int kenerl_w,
    const int pad_h, const int pad_w, const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w, const int deform_group,
    at::Tensor grad_im);

void modulated_deformable_col2im_coord_cuda(
    const at::Tensor data_col, const at::Tensor data_im,
    const at::Tensor data_offset, const at::Tensor data_mask,
    const int batch_size, const int channels, const int height_im,
    const int width_im, const int height_col, const int width_col,
    const int kernel_h, const int kenerl_w, const int pad_h, const int pad_w,
    const int stride_h, const int stride_w, const int dilation_h,
    const int dilation_w, const int deform_group, at::Tensor grad_offset,
    at::Tensor grad_mask);

void deform_attn_cuda_forward(
    at::Tensor q, at::Tensor kv, at::Tensor offset, at::Tensor output,
    at::Tensor columns, at::Tensor attns, at::Tensor mask_ones, int kernel_h, int kernel_w, const int stride_h, const int stride_w,
    const int pad_h, const int pad_w, const int dilation_h,
    const int dilation_w, const int attn_head, const int deform_group, const int clip_size
    ){
  TORCH_CHECK(kv.is_contiguous(), "input tensor has to be contiguous");
  at::DeviceGuard guard(kv.device());

  const int batch = q.size(0);
  const int kv_channels = kv.size(2);
  const int channels = kv.size(2) / 2;
  const int height = kv.size(3);
  const int width = kv.size(4);
  const int area = height * width;

  const int attn_dim = channels / attn_head;
  const int attn_size = kernel_h * kernel_w;
  const float attn_scale = pow(attn_dim, -0.5);

  // resize inputs
  q = q.view({batch, 1, attn_head, attn_dim, area}).permute({0, 2, 4, 1, 3}) * attn_scale; // batch x attn_head x (height*width) x 1 x attn_dim
  offset = offset.view({batch, clip_size, offset.size(1) / clip_size, area}); // batch x clip_size x (deform_groupxattn_sizex2) x (heightxwidht)

  output = output.view({batch, attn_head, attn_dim, height, width}).zero_();

  // resize temporary columns and attns
  columns = at::zeros({clip_size, kv_channels * attn_size, area}, q.options());
  attns = at::zeros({attn_head, area, 1, clip_size * attn_size}, q.options());
  mask_ones = at::ones({deform_group * attn_size, area}, q.options()); // batch x clip_size x (deform_group*attn_size) x (heightxwidth)

  for (int b = 0; b < batch; b++) { // 0->2,1->2, or, 1->3,0->3 // todo: refer to deformable_im2col_cuda and use `im2col_step` to speed up
    // grid_sample q and k according to offset
    for (int n = 0; n < clip_size; n++) {
        modulated_deformable_im2col_cuda(
        kv[b/clip_size][(n+b)%clip_size], offset[b][n], mask_ones, 1, kv_channels, height, width, height,
        width, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w,
        dilation_h, dilation_w, deform_group, columns[n]);
    }

    columns = columns.view({clip_size, 2, attn_head, attn_dim, attn_size, area})
               .permute({1, 2, 5, 3, 0, 4}).flatten(4); // kv x attn_head x (height*width) x attn_dim x (clip_size*attn_size)

    // calculate attention, (attn_head x (height*width) x 1 x attn_dim) @ (attn_head x (height*width) x attn_dim x (clip_size*attn_size))
    attns = at::matmul(q[b], columns[0])
                                    .softmax(-1); // (attn_head x (height*width) x 1 x (clip_size*attn_size))
    // do attention
    output[b] = at::matmul(attns, columns[1].transpose(2, 3)) // (attn_head x (height*width) x 1 x attn_dim)
                            .transpose(1, 3).view({attn_head, attn_dim, height, width}); // (attn_head x attn_dim x height x width)

    // resize columns back for next batch
    columns = columns.view({2, attn_head, area, attn_dim, clip_size , attn_size})
                                    .permute({4, 0, 1, 3, 5, 2}) // clip_size x attn_head x attn_dim x attn_size x (height*width)
                                    .flatten(1, 3); // clip_size x (attn_head*attn_dim*attn_size) x (height*width)
  }

  output = output.view({batch, channels, height, width});
}

void deform_attn_cuda_backward(
    at::Tensor q, at::Tensor kv, at::Tensor offset,
    at::Tensor columns, at::Tensor attns, at::Tensor mask_ones, at::Tensor grad_attns, at::Tensor grad_mask_ones, at::Tensor grad_q, at::Tensor grad_kv,
    at::Tensor grad_offset, at::Tensor grad_output,
    int kernel_h, int kernel_w, int stride_h, int stride_w, int pad_h,
    int pad_w, int dilation_h, int dilation_w, int attn_head, int deform_group, int clip_size
    ){
  at::DeviceGuard guard(kv.device());

  const int batch = q.size(0);
  const int kv_channels = kv.size(2);
  const int channels = kv.size(2) / 2;
  const int height = kv.size(3);
  const int width = kv.size(4);
  const int area = height * width;

  const int attn_dim = channels / attn_head;
  const int attn_size = kernel_h * kernel_w;
  const float attn_scale = pow(attn_dim, -0.5);
//  // for PyTorch 1.10.1
//  const at::ScalarType dtype = kv.scalar_type();

  // resize inputs
  q = q.view({batch, 1, attn_head, attn_dim, area}).permute({0, 2, 4, 1, 3}) * attn_scale; // batch x attn_head x (height*width) x 1 x attn_dim
  offset = offset.view({batch, clip_size, offset.size(1) / clip_size, area}); // batch x clip_size x (deform_groupxattn_sizex2) x (heightxwidht)

  grad_q = grad_q.view({batch, 1, attn_head, attn_dim, area}).permute({0, 2, 4, 1, 3});
  grad_offset = grad_offset.view({batch, clip_size, grad_offset.size(1) / clip_size, area});
  grad_output = grad_output.view({batch, 1, attn_head, attn_dim, area}).permute({0, 2, 4, 1, 3});

  // resize temporary columns, attns and grad_attns (we further need grad_attns in backward propagation because attn@V are interdependent.
  columns = at::zeros({clip_size, kv_channels * attn_size, area}, q.options());
  attns = at::zeros({attn_head, area, 1, clip_size * attn_size}, q.options());
  mask_ones = at::ones({deform_group * attn_size, area}, q.options()); // (deform_group*attn_size) x (heightxwidth)
  grad_attns = at::zeros({attn_head, area, 1, clip_size * attn_size}, q.options());
  grad_mask_ones = at::zeros({deform_group * attn_size, area}, q.options()); // not returned


  for (int b = 0; b < batch; b++) {
    // recalculate columns and attns
    // grid_sample q and k according to offset
    for (int n = 0; n < clip_size; n++) {
        modulated_deformable_im2col_cuda(
        kv[b/clip_size][(n+b)%clip_size], offset[b][n], mask_ones, 1, kv_channels, height, width, height,
        width, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w,
        dilation_h, dilation_w, deform_group, columns[n]);
    }

    columns = columns.view({clip_size, 2, attn_head, attn_dim, attn_size, area})
               .permute({1, 2, 5, 3, 0, 4}).flatten(4); // kv x attn_head x (height*width) x attn_dim x (clip_size*attn_size)

    // calculate attention, (attn_head x (height*width) x 1 x attn_dim) @ (attn_head x (height*width) x attn_dim x (clip_size*attn_size))
    attns = at::matmul(q[b], columns[0])
                                    .softmax(-1); // (attn_head x (height*width) x 1 x (clip_size*attn_size))

    // gradient w.r.t. attns, (attn_head x (height*width) x 1 x attn_dim) @ (attn_head x (height*width) x attn_dim x (clip_size*attn_size))
    grad_attns = at::matmul(grad_output[b], columns[1]); // (attn_head x (height*width) x 1 x (clip_size*attn_size))

    // gradient w.r.t. sampled_v, (attn_head x (height*width) x attn_dim x 1) @ (attn_head x (height*width) x 1 x (clip_size*attn_size))
    columns[1] = at::matmul(grad_output[b].transpose(2, 3), attns); // (attn_head x (height*width) x attn_dim x (clip_size*attn_size))

    // gradient w.r.t. attns_before_softmax
//     for PyTorch 1.9.1
    grad_attns = at::_softmax_backward_data(grad_attns, attns, -1, grad_attns); // todo: it seems pt191 has different interface as pt110
//    // for PyTorch 1.10.1
//    grad_attns = at::_softmax_backward_data(grad_attns, attns, -1, dtype);

    // gradient w.r.t. q, (attn_head x (height*width) x 1 x (clip_size*attn_size)) @ (attn_head x (height*width) x (clip_size*attn_size) x attn_dim)
    grad_q[b] = at::matmul(grad_attns, columns[0].transpose(2, 3)) * attn_scale; // (attn_head x (height*width) x 1 x attn_dim)

    // gradient w.r.t. sampled_k, (attn_head x (height*width) x attn_dim x 1) @ (attn_head x (height*width) x 1 x (clip_size*attn_size))
    columns[0] = at::matmul(q[b].transpose(2, 3), grad_attns) * attn_scale; // (attn_head x (height*width) x attn_dim x (clip_size*attn_size))

    columns = columns.view({2, attn_head, area, attn_dim, clip_size, attn_size})
                            .permute({4, 0, 1, 3, 5, 2}) // clip_size x 2 x attn_head x attn_dim x attn_size x (height*width)
                            .flatten(1, 4); // clip_size x (2*attn_head*attn_dim*attn_size) x (height*width)

    for (int n = 0; n < clip_size; n++) {
        // gradient w.r.t. input coordinate data (grad_offset and grad_mask_ones)
        modulated_deformable_col2im_coord_cuda(
            columns[n], kv[b/clip_size][(n+b)%clip_size], offset[b][n], mask_ones, 1, kv_channels, height, width,
            height, width, kernel_h, kernel_w, pad_h, pad_w, stride_h,
            stride_w, dilation_h, dilation_w, deform_group, grad_offset[b][n],
            grad_mask_ones);

        // gradient w.r.t. kv
        modulated_deformable_col2im_cuda(
            columns[n], offset[b][n], mask_ones, 1, kv_channels, height, width, height,
            width, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w,
            dilation_h, dilation_w, deform_group, grad_kv[b/clip_size][(n+b)%clip_size]); // the grad is accumulated
    }
  }

  // resize gradidents back
  grad_q = grad_q.transpose(2, 4).view({batch, channels, height, width}); // batch x (attn_headxattn_dim) x height x width
  grad_offset = grad_offset.flatten(1, 2);
  grad_output = grad_output.permute({0, 1, 4, 3, 2}).view({batch, channels, height, width});
}
