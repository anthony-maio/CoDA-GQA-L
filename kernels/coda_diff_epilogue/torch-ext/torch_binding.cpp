/*
 * PyTorch C++ bindings for fused differential attention epilogue.
 *
 * Registers the kernel as a torch custom op so it can be called from Python
 * and is compatible with torch.compile (via register_fake).
 */

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

// Forward declarations from diff_epilogue.cu
extern "C" {
void diff_epilogue_forward_bf16(
    __nv_bfloat16* output,
    const __nv_bfloat16* out_sig,
    const __nv_bfloat16* out_noise,
    const __nv_bfloat16* lam,
    const __nv_bfloat16* weight,
    int num_rows, int head_dim, float eps,
    cudaStream_t stream);

void diff_epilogue_forward_fp16(
    __half* output,
    const __half* out_sig,
    const __half* out_noise,
    const __half* lam,
    const __half* weight,
    int num_rows, int head_dim, float eps,
    cudaStream_t stream);

void diff_epilogue_forward_fp32(
    float* output,
    const float* out_sig,
    const float* out_noise,
    const float* lam,
    const float* weight,
    int num_rows, int head_dim, float eps,
    cudaStream_t stream);
}


void diff_epilogue_forward(
    torch::Tensor& output,
    const torch::Tensor& out_sig,
    const torch::Tensor& out_noise,
    const torch::Tensor& lam,
    const torch::Tensor& weight,
    double eps
) {
    TORCH_CHECK(out_sig.is_cuda(), "out_sig must be a CUDA tensor");
    TORCH_CHECK(out_noise.is_cuda(), "out_noise must be a CUDA tensor");
    TORCH_CHECK(lam.is_cuda(), "lam must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(output.is_cuda(), "output must be a CUDA tensor");

    TORCH_CHECK(out_sig.sizes() == out_noise.sizes(),
                "out_sig and out_noise must have the same shape");
    TORCH_CHECK(out_sig.sizes() == output.sizes(),
                "output must have the same shape as out_sig");

    // out_sig/out_noise: (B, H, Lq, Dh) flattened to (B*H*Lq, Dh)
    const int ndim = out_sig.dim();
    const int head_dim = out_sig.size(ndim - 1);
    const int num_rows = out_sig.numel() / head_dim;

    TORCH_CHECK(weight.numel() == head_dim,
                "weight must have head_dim elements, got ", weight.numel());
    TORCH_CHECK(lam.numel() == num_rows,
                "lam must have B*H*Lq elements, got ", lam.numel());

    const at::cuda::CUDAGuard device_guard(out_sig.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    auto sig_contig = out_sig.contiguous();
    auto noise_contig = out_noise.contiguous();
    auto lam_contig = lam.contiguous();
    auto weight_contig = weight.contiguous();

    if (out_sig.scalar_type() == at::kBFloat16) {
        diff_epilogue_forward_bf16(
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(sig_contig.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(noise_contig.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(lam_contig.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(weight_contig.data_ptr()),
            num_rows, head_dim, static_cast<float>(eps), stream
        );
    } else if (out_sig.scalar_type() == at::kHalf) {
        diff_epilogue_forward_fp16(
            reinterpret_cast<__half*>(output.data_ptr()),
            reinterpret_cast<const __half*>(sig_contig.data_ptr()),
            reinterpret_cast<const __half*>(noise_contig.data_ptr()),
            reinterpret_cast<const __half*>(lam_contig.data_ptr()),
            reinterpret_cast<const __half*>(weight_contig.data_ptr()),
            num_rows, head_dim, static_cast<float>(eps), stream
        );
    } else if (out_sig.scalar_type() == at::kFloat) {
        diff_epilogue_forward_fp32(
            static_cast<float*>(output.data_ptr()),
            static_cast<const float*>(sig_contig.data_ptr()),
            static_cast<const float*>(noise_contig.data_ptr()),
            static_cast<const float*>(lam_contig.data_ptr()),
            static_cast<const float*>(weight_contig.data_ptr()),
            num_rows, head_dim, static_cast<float>(eps), stream
        );
    } else {
        TORCH_CHECK(false, "Unsupported dtype: ", out_sig.scalar_type());
    }
}


// Register as torch ops
TORCH_LIBRARY(coda_kernels, m) {
    m.def("diff_epilogue_forward(Tensor! output, Tensor out_sig, Tensor out_noise, "
          "Tensor lam, Tensor weight, float eps) -> ()");
}

TORCH_LIBRARY_IMPL(coda_kernels, CUDA, m) {
    m.impl("diff_epilogue_forward", &diff_epilogue_forward);
}
