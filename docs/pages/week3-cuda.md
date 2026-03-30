# Week 3: CUDA -- GPU Programming {#page_week3}

## 1. GPU Architecture

GPUs are massively parallel processors designed for throughput, not latency. While a CPU has a few powerful cores optimized for sequential execution, a GPU has thousands of simpler cores optimized for parallel execution.

### CPU vs. GPU Philosophy

| Aspect | CPU | GPU |
|--------|-----|-----|
| Cores | 4-64 complex cores | 1000s of simple cores |
| Clock speed | 3-5 GHz | 1-2 GHz |
| Cache | Large (MB) | Small (KB per SM) |
| Control | Sophisticated branch prediction | Simple in-order |
| Strength | Latency | Throughput |

```
CPU: Few powerful cores          GPU: Many simple cores
+---------------------+          +---------------------------------+
| +-----+ +-----+    |          | -------- -------- --------     |
| |Core | |Core |    |          | -------- -------- --------     |
| |  0  | |  1  |    |          | -------- -------- --------     |
| +-----+ +-----+    |          | -------- -------- --------     |
| +-----+ +-----+    |          |   SM 0     SM 1     SM 2   ... |
| |Core | |Core |    |          +---------------------------------+
| |  2  | |  3  |    |
| +-----+ +-----+    |
+---------------------+
```

### NVIDIA GPU Hierarchy

**Streaming Multiprocessors (SMs)**: Independent processing units, each containing:
- CUDA cores (32-128 per SM)
- Shared memory / L1 cache
- Registers
- Warp schedulers

**Warps**: Groups of 32 threads that execute in lockstep (SIMT: Single Instruction, Multiple Threads).

---

## 2. CUDA Programming Model

### Kernels and Threads

A **kernel** is a function that runs on the GPU. When launched, it executes across many parallel threads.

```cpp
// Kernel definition (runs on GPU)
__global__ void add(float* a, float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

// Host code (runs on CPU)
int main() {
    // ... allocate and initialize ...

    // Launch kernel with 256 threads per block
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    add<<<blocks, threads>>>(d_a, d_b, d_c, n);

    cudaDeviceSynchronize();  // Wait for GPU to finish
}
```

### Thread Hierarchy

Threads are organized in a two-level hierarchy:

```
Grid (all threads for one kernel launch)
+-- Block 0
|   +-- Thread 0
|   +-- Thread 1
|   +-- ...
+-- Block 1
|   +-- Thread 0
|   +-- Thread 1
|   +-- ...
+-- ...
```

**Built-in variables**:
| Variable | Meaning |
|----------|---------|
| `threadIdx.x` | Thread index within block (0 to blockDim.x-1) |
| `blockIdx.x` | Block index within grid |
| `blockDim.x` | Threads per block |
| `gridDim.x` | Blocks in grid |

**Global thread ID**: `i = blockIdx.x * blockDim.x + threadIdx.x`

### Launch Configuration

```cpp
kernel<<<numBlocks, threadsPerBlock>>>(args...);
```

**Rules of thumb**:
- `threadsPerBlock`: Multiple of 32 (warp size), typically 128-512
- `numBlocks`: Enough to cover your data: `(n + threadsPerBlock - 1) / threadsPerBlock`

For 2D problems:
```cpp
dim3 block(16, 16);        // 256 threads per block
dim3 grid((n+15)/16, (m+15)/16);
kernel<<<grid, block>>>(...);

// Inside kernel:
int col = blockIdx.x * blockDim.x + threadIdx.x;
int row = blockIdx.y * blockDim.y + threadIdx.y;
```

---

## 3. Memory Hierarchy

GPU memory is hierarchical; choosing the right memory type is critical for performance.

```
+---------------------------------------------------------+
|                    Global Memory                        |
|                   (Large, slow: ~500 GB/s)              |
+---------------------------------------------------------+
        up                                      up
        |                                      |
+-------+-------+                    +---------+---------+
|     SM 0      |                    |       SM 1        |
| +-----------+ |                    |  +-----------+    |
| |  Shared   | |                    |  |  Shared   |    |
| |  Memory   | | (~100 TB/s)        |  |  Memory   |    |
| |  (48 KB)  | |                    |  |  (48 KB)  |    |
| +-----------+ |                    |  +-----------+    |
| +-----------+ |                    |  +-----------+    |
| | Registers | | (fastest)          |  | Registers |    |
| +-----------+ |                    |  +-----------+    |
+---------------+                    +-------------------+
```

### Memory Types

| Memory | Scope | Speed | Size | Lifetime |
|--------|-------|-------|------|----------|
| Registers | Thread | Fastest | ~255 per thread | Thread |
| Shared | Block | ~100 TB/s | 48-164 KB per SM | Block |
| Global | Grid | ~500-900 GB/s | GBs | Application |
| Constant | Grid | Cached | 64 KB | Application |

### Global Memory

Allocated on host, accessible by all threads:

```cpp
float* d_data;
cudaMalloc(&d_data, n * sizeof(float));        // Allocate
cudaMemcpy(d_data, h_data, n * sizeof(float),  // Copy H->D
           cudaMemcpyHostToDevice);
cudaMemcpy(h_data, d_data, n * sizeof(float),  // Copy D->H
           cudaMemcpyDeviceToHost);
cudaFree(d_data);                               // Free
```

### Shared Memory

Fast, on-chip memory shared by threads in a block:

```cpp
__global__ void reduce(float* input, float* output, int n) {
    __shared__ float sdata[256];  // Declare shared memory

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;

    // Load from global to shared
    sdata[tid] = (i < n) ? input[i] : 0;
    __syncthreads();  // Barrier: wait for all threads

    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write result
    if (tid == 0) output[blockIdx.x] = sdata[0];
}
```

**Key points**:
- Declare with `__shared__`
- Use `__syncthreads()` to synchronize threads within a block
- Bank conflicts can reduce performance (32 banks, consecutive addresses)

---

## 4. Common Patterns

### Map (Element-wise Operations)

Each thread processes one element:

```cpp
__global__ void scale(float* v, int n, float alpha) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        v[i] *= alpha;
    }
}
```

### Reduction

Combine all elements (sum, max, etc.):

```cpp
__global__ void dot_product(const float* x, const float* y, float* result, int n) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;

    // Each thread computes partial product
    sdata[tid] = (i < n) ? x[i] * y[i] : 0;
    __syncthreads();

    // Tree reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    // Atomic add to global result
    if (tid == 0) atomicAdd(result, sdata[0]);
}
```

**Note**: `atomicAdd` for doubles requires compute capability 6.0+ or a CAS-based implementation:

```cpp
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
```

### Stencil (Neighbors)

Each element depends on neighbors (e.g., finite differences):

```cpp
__global__ void stencil_1d(float* in, float* out, int n) {
    __shared__ float s[BLOCK + 2];  // Include halo

    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int lid = threadIdx.x + 1;  // Offset for halo

    // Load center
    s[lid] = (gid < n) ? in[gid] : 0;

    // Load halos
    if (threadIdx.x == 0 && gid > 0)
        s[0] = in[gid - 1];
    if (threadIdx.x == blockDim.x - 1 && gid < n - 1)
        s[lid + 1] = in[gid + 1];

    __syncthreads();

    // Compute stencil
    if (gid > 0 && gid < n - 1) {
        out[gid] = 0.25f * (s[lid-1] + 2*s[lid] + s[lid+1]);
    }
}
```

---

## 5. Matrix Operations on GPU

### Matrix-Vector Multiplication

Each thread computes one element of the output:

```cpp
__global__ void matvec(const double* A, const double* x, double* y,
                       int rows, int cols) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows) {
        double sum = 0;
        for (int j = 0; j < cols; ++j) {
            sum += A[i * cols + j] * x[j];
        }
        y[i] = sum;
    }
}
```

### Matrix-Matrix Multiplication

Naive version (each thread computes one element of C):

```cpp
__global__ void matmul_naive(const double* A, const double* B, double* C,
                             int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        double sum = 0;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}
```

**Tiled version** using shared memory (much faster):

```cpp
#define TILE 16

__global__ void matmul_tiled(const double* A, const double* B, double* C,
                             int M, int K, int N) {
    __shared__ double As[TILE][TILE];
    __shared__ double Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    double sum = 0;

    // Loop over tiles
    for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
        // Load tiles into shared memory
        int a_col = t * TILE + threadIdx.x;
        int b_row = t * TILE + threadIdx.y;

        As[threadIdx.y][threadIdx.x] = (row < M && a_col < K) ?
                                        A[row * K + a_col] : 0;
        Bs[threadIdx.y][threadIdx.x] = (b_row < K && col < N) ?
                                        B[b_row * N + col] : 0;

        __syncthreads();

        // Compute partial dot product
        for (int k = 0; k < TILE; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}
```

**Why tiling helps**:
- Without tiling: Each element of A and B is read \f$N\f$ times from global memory
- With tiling: Each element is read once per tile into shared memory, then reused

---

## 6. Performance Optimization

### Coalesced Memory Access

Threads in a warp should access consecutive memory addresses:

```cpp
// Good: Coalesced (thread i accesses element i)
int i = blockIdx.x * blockDim.x + threadIdx.x;
float val = data[i];

// Bad: Strided (thread i accesses element i*stride)
float val = data[i * stride];  // Much slower
```

### Occupancy

**Occupancy** = active warps / maximum warps per SM

Higher occupancy helps hide memory latency. Limited by:
- Registers per thread
- Shared memory per block
- Threads per block

Use `cudaOccupancyMaxPotentialBlockSize` to find optimal configuration.

### Avoiding Warp Divergence

Threads in a warp execute the same instruction. If threads take different branches, both paths execute serially:

```cpp
// Bad: Divergent
if (threadIdx.x % 2 == 0) {
    // Half the warp does this
} else {
    // Half does this
}

// Better: Whole warps take same branch
if (threadIdx.x < 16) {
    // First half-warp
} else {
    // Second half-warp
}
```

### Use the Profiler

NVIDIA tools help identify bottlenecks:
- `nvprof` / `nsys` -- Timeline profiler
- `ncu` (Nsight Compute) -- Kernel analysis
- `cuda-memcheck` -- Memory error detection

```bash
nvprof ./my_program
nsys profile ./my_program
ncu --set full ./my_program
```

---

## 7. Our Library's CUDA Interface

The `num::cuda` namespace provides low-level operations:

```cpp
namespace num::cuda {
    real* alloc(idx n);                           // cudaMalloc wrapper
    void free(real* ptr);                         // cudaFree wrapper
    void to_device(real* dst, const real* src, idx n);
    void to_host(real* dst, const real* src, idx n);

    void scale(real* v, idx n, real alpha);
    void add(const real* x, const real* y, real* z, idx n);
    void axpy(real alpha, const real* x, real* y, idx n);
    real dot(const real* x, const real* y, idx n);

    void matvec(const real* A, const real* x, real* y, idx rows, idx cols);
    void matmul(const real* A, const real* B, real* C, idx m, idx k, idx n);
}
```

High-level interface with automatic memory management:

```cpp
num::Vector x(1000, 1.0);
num::Vector y(1000, 2.0);

x.to_gpu();  // Transfer to device
y.to_gpu();

num::axpy(2.0, x, y, num::Exec::gpu);  // y = 2*x + y on GPU

y.to_cpu();  // Transfer back
```

---

## 8. CPU vs GPU: When to Use What

| Use CPU | Use GPU |
|---------|---------|
| Small data (< 10K elements) | Large data (> 100K elements) |
| Complex branching logic | Regular, data-parallel ops |
| Sequential algorithms | Embarrassingly parallel |
| Quick prototyping | Production performance |

**Kernel launch overhead**: ~5-20 mus. For small operations, this dominates.

From our benchmarks:

| Operation | Crossover Point |
|-----------|-----------------|
| Dot product | GPU never wins (memory-bound, launch overhead dominates) |
| AXPY | ~64K elements |
| Matvec | ~256x256 |
| Matmul | GPU always wins (compute-bound) |

---

## Exercises

1. Write a CUDA kernel that computes \f$\mathbf{z} = \mathbf{x} \cdot \mathbf{y}\f$ (element-wise product).

2. Explain why the tiled matrix multiplication is faster. Calculate the reduction in global memory accesses for a 1024x1024 multiplication with TILE=32.

3. A kernel uses 64 registers per thread and 48 KB shared memory. If the GPU has 65536 registers and 96 KB shared memory per SM, what is the maximum occupancy with 256 threads per block?

4. Identify the warp divergence in this code and fix it:
   ```cpp
   if (threadIdx.x > data[threadIdx.x]) {
       output[threadIdx.x] = 1;
   } else {
       output[threadIdx.x] = 0;
   }
   ```

5. Profile the library's `matmul` on GPU using `nvprof`. What is the achieved memory bandwidth? How does it compare to the theoretical peak?

---

## References

- Kirk & Hwu, *Programming Massively Parallel Processors*
- NVIDIA, *CUDA C++ Programming Guide*
- NVIDIA, *CUDA C++ Best Practices Guide*
- Harris, "Optimizing Parallel Reduction in CUDA"
