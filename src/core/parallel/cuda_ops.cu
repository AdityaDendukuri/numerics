#include "core/parallel/cuda_ops.hpp"
#include <cuda_runtime.h>

namespace num::cuda {

namespace {
constexpr int BLOCK = 256;

#if __CUDA_ARCH__ < 600
__device__ double atomicAddDouble(double* addr, double val) {
    unsigned long long* addr_ull = (unsigned long long*)addr;
    unsigned long long old = *addr_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(addr_ull, assumed,
            __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}
#else
__device__ double atomicAddDouble(double* addr, double val) {
    return atomicAdd(addr, val);
}
#endif

__global__ void k_scale(real* v, idx n, real alpha) {
    idx i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) v[i] *= alpha;
}

__global__ void k_add(const real* x, const real* y, real* z, idx n) {
    idx i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) z[i] = x[i] + y[i];
}

__global__ void k_axpy(real alpha, const real* x, real* y, idx n) {
    idx i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += alpha * x[i];
}

__global__ void k_dot(const real* x, const real* y, real* result, idx n) {
    __shared__ real sdata[BLOCK];
    idx tid = threadIdx.x;
    idx i = blockIdx.x * blockDim.x + tid;
    sdata[tid] = (i < n) ? x[i] * y[i] : 0;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) atomicAddDouble(result, sdata[0]);
}

__global__ void k_matvec(const real* A, const real* x, real* y, idx rows, idx cols) {
    idx i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows) {
        real sum = 0;
        for (idx j = 0; j < cols; ++j) sum += A[i * cols + j] * x[j];
        y[i] = sum;
    }
}

__global__ void k_matmul(const real* A, const real* B, real* C, idx m, idx k, idx n) {
    idx row = blockIdx.y * blockDim.y + threadIdx.y;
    idx col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m && col < n) {
        real sum = 0;
        for (idx i = 0; i < k; ++i) sum += A[row * k + i] * B[i * n + col];
        C[row * n + col] = sum;
    }
}

__global__ void k_thomas_batched(const real* a, const real* b, const real* c,
                                  const real* d, real* x,
                                  real* b_work, real* d_work,
                                  idx n, idx batch_size) {
    idx batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch_idx >= batch_size) return;
    idx off_sub = batch_idx * (n - 1);
    idx off_main = batch_idx * n;
    for (idx i = 0; i < n; ++i) {
        b_work[off_main + i] = b[off_main + i];
        d_work[off_main + i] = d[off_main + i];
    }
    for (idx i = 1; i < n; ++i) {
        real w = a[off_sub + i - 1] / b_work[off_main + i - 1];
        b_work[off_main + i] -= w * c[off_sub + i - 1];
        d_work[off_main + i] -= w * d_work[off_main + i - 1];
    }
    x[off_main + n - 1] = d_work[off_main + n - 1] / b_work[off_main + n - 1];
    for (idx i = n - 1; i > 0; --i) {
        x[off_main + i - 1] = (d_work[off_main + i - 1] - c[off_sub + i - 1] * x[off_main + i])
                              / b_work[off_main + i - 1];
    }
}
} // anon namespace

real* alloc(idx n) { real* ptr; cudaMalloc(&ptr, n * sizeof(real)); return ptr; }
void free(real* ptr) { cudaFree(ptr); }
void to_device(real* dst, const real* src, idx n) { cudaMemcpy(dst, src, n * sizeof(real), cudaMemcpyHostToDevice); }
void to_host(real* dst, const real* src, idx n) { cudaMemcpy(dst, src, n * sizeof(real), cudaMemcpyDeviceToHost); }
void scale(real* v, idx n, real alpha) { k_scale<<<(n + BLOCK - 1) / BLOCK, BLOCK>>>(v, n, alpha); }
void add(const real* x, const real* y, real* z, idx n) { k_add<<<(n + BLOCK - 1) / BLOCK, BLOCK>>>(x, y, z, n); }
void axpy(real alpha, const real* x, real* y, idx n) { k_axpy<<<(n + BLOCK - 1) / BLOCK, BLOCK>>>(alpha, x, y, n); }

real dot(const real* x, const real* y, idx n) {
    real* d_result; cudaMalloc(&d_result, sizeof(real)); cudaMemset(d_result, 0, sizeof(real));
    k_dot<<<(n + BLOCK - 1) / BLOCK, BLOCK>>>(x, y, d_result, n);
    real result; cudaMemcpy(&result, d_result, sizeof(real), cudaMemcpyDeviceToHost);
    cudaFree(d_result); return result;
}

void matvec(const real* A, const real* x, real* y, idx rows, idx cols) {
    k_matvec<<<(rows + BLOCK - 1) / BLOCK, BLOCK>>>(A, x, y, rows, cols);
}

void matmul(const real* A, const real* B, real* C, idx m, idx k, idx n) {
    dim3 block(16, 16), grid((n + 15) / 16, (m + 15) / 16);
    k_matmul<<<grid, block>>>(A, B, C, m, k, n);
}

void thomas_batched(const real* a, const real* b, const real* c,
                    const real* d, real* x, idx n, idx batch_size) {
    real* b_work; real* d_work;
    cudaMalloc(&b_work, batch_size * n * sizeof(real));
    cudaMalloc(&d_work, batch_size * n * sizeof(real));
    k_thomas_batched<<<(batch_size + BLOCK - 1) / BLOCK, BLOCK>>>(
        a, b, c, d, x, b_work, d_work, n, batch_size);
    cudaFree(b_work); cudaFree(d_work);
}

} // namespace num::cuda
