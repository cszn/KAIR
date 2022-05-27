// modify from
// https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/blob/mmdetection/mmdet/ops/dcn/src/deform_conv_cuda.c

#include <torch/extension.h>
#include <ATen/DeviceGuard.h>

#include <cmath>
#include <vector>

#define WITH_CUDA  // always use cuda
#ifdef WITH_CUDA

void deform_attn_cuda_forward(
    at::Tensor q, at::Tensor kv, at::Tensor offset, at::Tensor output,
    at::Tensor columns, at::Tensor attns, at::Tensor mask_ones, int kernel_h, int kernel_w, const int stride_h, const int stride_w,
    const int pad_h, const int pad_w, const int dilation_h,
    const int dilation_w, const int attn_head, const int deform_group, const int clip_size
    );

void deform_attn_cuda_backward(
    at::Tensor q, at::Tensor kv, at::Tensor offset,
    at::Tensor columns, at::Tensor attns, at::Tensor mask_ones, at::Tensor grad_attns, at::Tensor grad_mask_ones, at::Tensor grad_q, at::Tensor grad_kv,
    at::Tensor grad_offset, at::Tensor grad_output,
    int kernel_h, int kernel_w, int stride_h, int stride_w, int pad_h,
    int pad_w, int dilation_h, int dilation_w, int attn_head, int deform_group, int clip_size
    );
#endif

void deform_attn_forward(
    at::Tensor q, at::Tensor kv, at::Tensor offset, at::Tensor output,
    at::Tensor columns, at::Tensor attns, at::Tensor mask_ones, int kernel_h, int kernel_w, const int stride_h, const int stride_w,
    const int pad_h, const int pad_w, const int dilation_h,
    const int dilation_w, const int attn_head, const int deform_group, const int clip_size
    ) {
  if (q.device().is_cuda()) {
#ifdef WITH_CUDA
    return deform_attn_cuda_forward(q, kv,
        offset, output, columns, attns, mask_ones, kernel_h, kernel_w, stride_h,
        stride_w, pad_h, pad_w, dilation_h, dilation_w, attn_head, deform_group, clip_size);
#else
    AT_ERROR("modulated deform attn is not compiled with GPU support");
#endif
  }
  AT_ERROR("modulated deform attn is not implemented on CPU");
}

void deform_attn_backward(
    at::Tensor q, at::Tensor kv, at::Tensor offset, at::Tensor columns,
    at::Tensor attns, at::Tensor mask_ones, at::Tensor grad_attns, at::Tensor grad_mask_ones, at::Tensor grad_q, at::Tensor grad_kv,
    at::Tensor grad_offset, at::Tensor grad_output,
    int kernel_h, int kernel_w, int stride_h, int stride_w, int pad_h,
    int pad_w, int dilation_h, int dilation_w, int attn_head, int deform_group, int clip_size
    ) {
  if (q.device().is_cuda()) {
#ifdef WITH_CUDA
    return deform_attn_cuda_backward(q, kv,
        offset, columns, attns, mask_ones, grad_attns, grad_mask_ones, grad_q, grad_kv, grad_offset,
        grad_output, kernel_h, kernel_w, stride_h, stride_w,
        pad_h, pad_w, dilation_h, dilation_w, attn_head, deform_group, clip_size);
#else
    AT_ERROR("modulated deform attn is not compiled with GPU support");
#endif
  }
  AT_ERROR("modulated deform attn is not implemented on CPU");
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("deform_attn_forward",
        &deform_attn_forward,
        "deform attn forward");
  m.def("deform_attn_backward",
        &deform_attn_backward,
        "deform attn backward");
}
