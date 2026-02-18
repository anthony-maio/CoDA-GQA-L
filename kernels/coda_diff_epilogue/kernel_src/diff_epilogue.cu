/*
 * Fused Differential Attention Epilogue Kernel
 *
 * Fuses three operations into a single global memory pass:
 *   1. Subtraction:  diff = out_sig - lam * out_noise
 *   2. RMSNorm:      rms_inv = rsqrt(mean(diff^2) + eps)
 *   3. Scaling:       output = diff * rms_inv * weight
 *
 * Without fusion, PyTorch materializes two intermediate tensors
 * (the subtraction result and the normalized result), causing
 * three global memory round-trips. This kernel does one read of
 * sig+noise and one write of the final output.
 *
 * Layout: (B, H, Lq, Dh) -- one block per (b, h, q) row of Dh elements.
 *
 * Supports: BF16 (sm_80+), FP16 (sm_75+), FP32
 * Target: H100 (sm_90), A100 (sm_80), T4 (sm_75)
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cmath>

// ---------------------------------------------------------------------------
// Type conversion helpers (required for PyTorch's -D__CUDA_NO_HALF_OPERATORS__)
// ---------------------------------------------------------------------------

__device__ __forceinline__ float to_float(float x) { return x; }
__device__ __forceinline__ float to_float(__half x) { return __half2float(x); }
__device__ __forceinline__ float to_float(__nv_bfloat16 x) { return __bfloat162float(x); }

__device__ __forceinline__ float from_float_f(float x) { return x; }
__device__ __forceinline__ __half from_float_h(float x) { return __float2half(x); }
__device__ __forceinline__ __nv_bfloat16 from_float_b(float x) { return __float2bfloat16(x); }

// ---------------------------------------------------------------------------
// Warp and block reductions
// ---------------------------------------------------------------------------

constexpr int WARP_SIZE = 32;

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ float block_reduce_sum(float val, float* shared) {
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;

    val = warp_reduce_sum(val);

    if (lane == 0) shared[wid] = val;
    __syncthreads();

    val = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared[lane] : 0.0f;
    if (wid == 0) val = warp_reduce_sum(val);

    return val;
}

// ---------------------------------------------------------------------------
// BF16 vectorized kernel (sm_80+: A100, H100)
// ---------------------------------------------------------------------------

__global__ void diff_epilogue_bf16_vec(
    __nv_bfloat16* __restrict__ output,          // (B*H*Lq, Dh)
    const __nv_bfloat16* __restrict__ out_sig,   // (B*H*Lq, Dh)
    const __nv_bfloat16* __restrict__ out_noise, // (B*H*Lq, Dh)
    const __nv_bfloat16* __restrict__ lam,       // (B*H*Lq,)
    const __nv_bfloat16* __restrict__ weight,    // (Dh,)
    const int head_dim,
    const float eps
) {
    extern __shared__ char smem[];
    float* shared = reinterpret_cast<float*>(smem);

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;

    // Each row is one (batch, head, query_pos) slice
    const __nv_bfloat16* row_sig   = out_sig   + row * head_dim;
    const __nv_bfloat16* row_noise = out_noise  + row * head_dim;
    __nv_bfloat16* row_out = output + row * head_dim;

    // Load per-row lambda (single scalar)
    const float lambda_val = __bfloat162float(lam[row]);

    // --- Phase 1: Compute diff and accumulate sum-of-squares ---
    // Use bf16x2 vectorized loads (2 elements per 32-bit load)
    const int vec_dim = head_dim / 2;
    const __nv_bfloat162* vec_sig   = reinterpret_cast<const __nv_bfloat162*>(row_sig);
    const __nv_bfloat162* vec_noise = reinterpret_cast<const __nv_bfloat162*>(row_noise);

    float sum_sq = 0.0f;

    #pragma unroll 4
    for (int i = tid; i < vec_dim; i += stride) {
        __nv_bfloat162 s = vec_sig[i];
        __nv_bfloat162 n = vec_noise[i];

        float s0 = __bfloat162float(s.x);
        float s1 = __bfloat162float(s.y);
        float n0 = __bfloat162float(n.x);
        float n1 = __bfloat162float(n.y);

        float d0 = s0 - lambda_val * n0;
        float d1 = s1 - lambda_val * n1;

        sum_sq += d0 * d0 + d1 * d1;
    }

    // Handle odd head_dim (shouldn't happen in practice, Dh is always even)
    if (head_dim % 2 == 1 && tid == 0) {
        float s = __bfloat162float(row_sig[head_dim - 1]);
        float n = __bfloat162float(row_noise[head_dim - 1]);
        float d = s - lambda_val * n;
        sum_sq += d * d;
    }

    // Block-wide reduction
    sum_sq = block_reduce_sum(sum_sq, shared);

    // Compute RMS inverse
    __shared__ float s_rms_inv;
    if (tid == 0) {
        float mean_sq = sum_sq / static_cast<float>(head_dim);
        s_rms_inv = rsqrtf(mean_sq + eps);
    }
    __syncthreads();

    const float rms_inv = s_rms_inv;

    // --- Phase 2: Normalize and write with bf16x2 vectorized stores ---
    const __nv_bfloat162* vec_weight = reinterpret_cast<const __nv_bfloat162*>(weight);
    __nv_bfloat162* vec_out = reinterpret_cast<__nv_bfloat162*>(row_out);

    #pragma unroll 4
    for (int i = tid; i < vec_dim; i += stride) {
        __nv_bfloat162 s = vec_sig[i];
        __nv_bfloat162 n = vec_noise[i];
        __nv_bfloat162 w = vec_weight[i];

        float s0 = __bfloat162float(s.x);
        float s1 = __bfloat162float(s.y);
        float n0 = __bfloat162float(n.x);
        float n1 = __bfloat162float(n.y);
        float w0 = __bfloat162float(w.x);
        float w1 = __bfloat162float(w.y);

        float d0 = s0 - lambda_val * n0;
        float d1 = s1 - lambda_val * n1;

        __nv_bfloat162 result;
        result.x = __float2bfloat16(d0 * rms_inv * w0);
        result.y = __float2bfloat16(d1 * rms_inv * w1);
        vec_out[i] = result;
    }

    if (head_dim % 2 == 1 && tid == 0) {
        float s = __bfloat162float(row_sig[head_dim - 1]);
        float n = __bfloat162float(row_noise[head_dim - 1]);
        float w = __bfloat162float(weight[head_dim - 1]);
        float d = s - lambda_val * n;
        row_out[head_dim - 1] = __float2bfloat16(d * rms_inv * w);
    }
}

// ---------------------------------------------------------------------------
// FP16 vectorized kernel (sm_75+: T4, A100, H100)
// ---------------------------------------------------------------------------

__global__ void diff_epilogue_fp16_vec(
    __half* __restrict__ output,
    const __half* __restrict__ out_sig,
    const __half* __restrict__ out_noise,
    const __half* __restrict__ lam,
    const __half* __restrict__ weight,
    const int head_dim,
    const float eps
) {
    extern __shared__ char smem[];
    float* shared = reinterpret_cast<float*>(smem);

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;

    const __half* row_sig   = out_sig   + row * head_dim;
    const __half* row_noise = out_noise + row * head_dim;
    __half* row_out = output + row * head_dim;

    const float lambda_val = __half2float(lam[row]);

    const int vec_dim = head_dim / 2;
    const __half2* vec_sig   = reinterpret_cast<const __half2*>(row_sig);
    const __half2* vec_noise = reinterpret_cast<const __half2*>(row_noise);

    float sum_sq = 0.0f;

    #pragma unroll 4
    for (int i = tid; i < vec_dim; i += stride) {
        __half2 s = vec_sig[i];
        __half2 n = vec_noise[i];

        float s0 = __half2float(s.x);
        float s1 = __half2float(s.y);
        float n0 = __half2float(n.x);
        float n1 = __half2float(n.y);

        float d0 = s0 - lambda_val * n0;
        float d1 = s1 - lambda_val * n1;

        sum_sq += d0 * d0 + d1 * d1;
    }

    if (head_dim % 2 == 1 && tid == 0) {
        float s = __half2float(row_sig[head_dim - 1]);
        float n = __half2float(row_noise[head_dim - 1]);
        float d = s - lambda_val * n;
        sum_sq += d * d;
    }

    sum_sq = block_reduce_sum(sum_sq, shared);

    __shared__ float s_rms_inv;
    if (tid == 0) {
        s_rms_inv = rsqrtf(sum_sq / static_cast<float>(head_dim) + eps);
    }
    __syncthreads();

    const float rms_inv = s_rms_inv;

    const __half2* vec_weight = reinterpret_cast<const __half2*>(weight);
    __half2* vec_out = reinterpret_cast<__half2*>(row_out);

    #pragma unroll 4
    for (int i = tid; i < vec_dim; i += stride) {
        __half2 s = vec_sig[i];
        __half2 n = vec_noise[i];
        __half2 w = vec_weight[i];

        float s0 = __half2float(s.x);
        float s1 = __half2float(s.y);
        float n0 = __half2float(n.x);
        float n1 = __half2float(n.y);
        float w0 = __half2float(w.x);
        float w1 = __half2float(w.y);

        float d0 = s0 - lambda_val * n0;
        float d1 = s1 - lambda_val * n1;

        __half2 result;
        result.x = __float2half(d0 * rms_inv * w0);
        result.y = __float2half(d1 * rms_inv * w1);
        vec_out[i] = result;
    }

    if (head_dim % 2 == 1 && tid == 0) {
        float s = __half2float(row_sig[head_dim - 1]);
        float n = __half2float(row_noise[head_dim - 1]);
        float w = __half2float(weight[head_dim - 1]);
        float d = s - lambda_val * n;
        row_out[head_dim - 1] = __float2half(d * rms_inv * w);
    }
}

// ---------------------------------------------------------------------------
// FP32 kernel (fallback / debugging)
// ---------------------------------------------------------------------------

__global__ void diff_epilogue_fp32(
    float* __restrict__ output,
    const float* __restrict__ out_sig,
    const float* __restrict__ out_noise,
    const float* __restrict__ lam,
    const float* __restrict__ weight,
    const int head_dim,
    const float eps
) {
    extern __shared__ char smem[];
    float* shared = reinterpret_cast<float*>(smem);

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;

    const float* row_sig   = out_sig   + row * head_dim;
    const float* row_noise = out_noise + row * head_dim;
    float* row_out = output + row * head_dim;

    const float lambda_val = lam[row];

    float sum_sq = 0.0f;
    for (int i = tid; i < head_dim; i += stride) {
        float d = row_sig[i] - lambda_val * row_noise[i];
        sum_sq += d * d;
    }

    sum_sq = block_reduce_sum(sum_sq, shared);

    __shared__ float s_rms_inv;
    if (tid == 0) {
        s_rms_inv = rsqrtf(sum_sq / static_cast<float>(head_dim) + eps);
    }
    __syncthreads();

    const float rms_inv = s_rms_inv;

    for (int i = tid; i < head_dim; i += stride) {
        float d = row_sig[i] - lambda_val * row_noise[i];
        row_out[i] = d * rms_inv * weight[i];
    }
}

// ---------------------------------------------------------------------------
// C entry points (called from torch_binding.cpp)
// ---------------------------------------------------------------------------

// Compute optimal thread count for a given head_dim
static int get_threads(int head_dim, bool vectorized) {
    int elems = vectorized ? head_dim / 2 : head_dim;
    int threads = elems;
    if (threads > 1024) threads = 1024;
    if (threads < WARP_SIZE) threads = WARP_SIZE;
    // Round up to warp boundary
    threads = ((threads + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
    return threads;
}

// ---------------------------------------------------------------------------
// BF16 subtract-only kernel (identity head norm, no RMSNorm)
// Fuses: output = out_sig - lam * out_noise
// ---------------------------------------------------------------------------

__global__ void diff_sub_only_bf16_vec(
    __nv_bfloat16* __restrict__ output,
    const __nv_bfloat16* __restrict__ out_sig,
    const __nv_bfloat16* __restrict__ out_noise,
    const __nv_bfloat16* __restrict__ lam,
    const int head_dim
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;

    const __nv_bfloat16* row_sig   = out_sig   + row * head_dim;
    const __nv_bfloat16* row_noise = out_noise  + row * head_dim;
    __nv_bfloat16* row_out = output + row * head_dim;

    const float lambda_val = __bfloat162float(lam[row]);

    const int vec_dim = head_dim / 2;
    const __nv_bfloat162* vec_sig   = reinterpret_cast<const __nv_bfloat162*>(row_sig);
    const __nv_bfloat162* vec_noise = reinterpret_cast<const __nv_bfloat162*>(row_noise);
    __nv_bfloat162* vec_out = reinterpret_cast<__nv_bfloat162*>(row_out);

    #pragma unroll 4
    for (int i = tid; i < vec_dim; i += stride) {
        __nv_bfloat162 s = vec_sig[i];
        __nv_bfloat162 n = vec_noise[i];

        float s0 = __bfloat162float(s.x);
        float s1 = __bfloat162float(s.y);
        float n0 = __bfloat162float(n.x);
        float n1 = __bfloat162float(n.y);

        __nv_bfloat162 result;
        result.x = __float2bfloat16(s0 - lambda_val * n0);
        result.y = __float2bfloat16(s1 - lambda_val * n1);
        vec_out[i] = result;
    }

    if (head_dim % 2 == 1 && tid == 0) {
        float s = __bfloat162float(row_sig[head_dim - 1]);
        float n = __bfloat162float(row_noise[head_dim - 1]);
        row_out[head_dim - 1] = __float2bfloat16(s - lambda_val * n);
    }
}

// ---------------------------------------------------------------------------
// FP16 subtract-only kernel
// ---------------------------------------------------------------------------

__global__ void diff_sub_only_fp16_vec(
    __half* __restrict__ output,
    const __half* __restrict__ out_sig,
    const __half* __restrict__ out_noise,
    const __half* __restrict__ lam,
    const int head_dim
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;

    const __half* row_sig   = out_sig   + row * head_dim;
    const __half* row_noise = out_noise + row * head_dim;
    __half* row_out = output + row * head_dim;

    const float lambda_val = __half2float(lam[row]);

    const int vec_dim = head_dim / 2;
    const __half2* vec_sig   = reinterpret_cast<const __half2*>(row_sig);
    const __half2* vec_noise = reinterpret_cast<const __half2*>(row_noise);
    __half2* vec_out = reinterpret_cast<__half2*>(row_out);

    #pragma unroll 4
    for (int i = tid; i < vec_dim; i += stride) {
        __half2 s = vec_sig[i];
        __half2 n = vec_noise[i];

        float s0 = __half2float(s.x);
        float s1 = __half2float(s.y);
        float n0 = __half2float(n.x);
        float n1 = __half2float(n.y);

        __half2 result;
        result.x = __float2half(s0 - lambda_val * n0);
        result.y = __float2half(s1 - lambda_val * n1);
        vec_out[i] = result;
    }

    if (head_dim % 2 == 1 && tid == 0) {
        float s = __half2float(row_sig[head_dim - 1]);
        float n = __half2float(row_noise[head_dim - 1]);
        row_out[head_dim - 1] = __float2half(s - lambda_val * n);
    }
}

// ---------------------------------------------------------------------------
// FP32 subtract-only kernel
// ---------------------------------------------------------------------------

__global__ void diff_sub_only_fp32(
    float* __restrict__ output,
    const float* __restrict__ out_sig,
    const float* __restrict__ out_noise,
    const float* __restrict__ lam,
    const int head_dim
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;

    const float* row_sig   = out_sig   + row * head_dim;
    const float* row_noise = out_noise + row * head_dim;
    float* row_out = output + row * head_dim;

    const float lambda_val = lam[row];

    for (int i = tid; i < head_dim; i += stride) {
        row_out[i] = row_sig[i] - lambda_val * row_noise[i];
    }
}

// ---------------------------------------------------------------------------
// C entry points (called from torch_binding.cpp)
// ---------------------------------------------------------------------------

extern "C" {

void diff_epilogue_forward_bf16(
    __nv_bfloat16* output,
    const __nv_bfloat16* out_sig,
    const __nv_bfloat16* out_noise,
    const __nv_bfloat16* lam,
    const __nv_bfloat16* weight,
    int num_rows,
    int head_dim,
    float eps,
    cudaStream_t stream
) {
    int threads = get_threads(head_dim, true);
    size_t smem = ((threads + WARP_SIZE - 1) / WARP_SIZE) * sizeof(float);
    diff_epilogue_bf16_vec<<<num_rows, threads, smem, stream>>>(
        output, out_sig, out_noise, lam, weight, head_dim, eps
    );
}

void diff_epilogue_forward_fp16(
    __half* output,
    const __half* out_sig,
    const __half* out_noise,
    const __half* lam,
    const __half* weight,
    int num_rows,
    int head_dim,
    float eps,
    cudaStream_t stream
) {
    int threads = get_threads(head_dim, true);
    size_t smem = ((threads + WARP_SIZE - 1) / WARP_SIZE) * sizeof(float);
    diff_epilogue_fp16_vec<<<num_rows, threads, smem, stream>>>(
        output, out_sig, out_noise, lam, weight, head_dim, eps
    );
}

void diff_epilogue_forward_fp32(
    float* output,
    const float* out_sig,
    const float* out_noise,
    const float* lam,
    const float* weight,
    int num_rows,
    int head_dim,
    float eps,
    cudaStream_t stream
) {
    int threads = get_threads(head_dim, false);
    size_t smem = ((threads + WARP_SIZE - 1) / WARP_SIZE) * sizeof(float);
    diff_epilogue_fp32<<<num_rows, threads, smem, stream>>>(
        output, out_sig, out_noise, lam, weight, head_dim, eps
    );
}

// --- Subtract-only entry points (identity head norm) ---

void diff_sub_only_forward_bf16(
    __nv_bfloat16* output,
    const __nv_bfloat16* out_sig,
    const __nv_bfloat16* out_noise,
    const __nv_bfloat16* lam,
    int num_rows,
    int head_dim,
    cudaStream_t stream
) {
    int threads = get_threads(head_dim, true);
    diff_sub_only_bf16_vec<<<num_rows, threads, 0, stream>>>(
        output, out_sig, out_noise, lam, head_dim
    );
}

void diff_sub_only_forward_fp16(
    __half* output,
    const __half* out_sig,
    const __half* out_noise,
    const __half* lam,
    int num_rows,
    int head_dim,
    cudaStream_t stream
) {
    int threads = get_threads(head_dim, true);
    diff_sub_only_fp16_vec<<<num_rows, threads, 0, stream>>>(
        output, out_sig, out_noise, lam, head_dim
    );
}

void diff_sub_only_forward_fp32(
    float* output,
    const float* out_sig,
    const float* out_noise,
    const float* lam,
    int num_rows,
    int head_dim,
    cudaStream_t stream
) {
    int threads = get_threads(head_dim, false);
    diff_sub_only_fp32<<<num_rows, threads, 0, stream>>>(
        output, out_sig, out_noise, lam, head_dim
    );
}

}  // extern "C"
