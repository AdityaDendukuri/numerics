#include "core/parallel/cuda_ops.hpp"
#include <stdexcept>

namespace num::cuda {

[[noreturn]] static void no_cuda() {
    throw std::runtime_error("CUDA not available");
}

real* alloc(idx) { no_cuda(); }
void free(real*) { no_cuda(); }
void to_device(real*, const real*, idx) { no_cuda(); }
void to_host(real*, const real*, idx) { no_cuda(); }
void scale(real*, idx, real) { no_cuda(); }
void add(const real*, const real*, real*, idx) { no_cuda(); }
void axpy(real, const real*, real*, idx) { no_cuda(); }
real dot(const real*, const real*, idx) { no_cuda(); }
void matvec(const real*, const real*, real*, idx, idx) { no_cuda(); }
void matmul(const real*, const real*, real*, idx, idx, idx) { no_cuda(); }
void thomas_batched(const real*, const real*, const real*, const real*, real*, idx, idx) { no_cuda(); }

} // namespace num::cuda
