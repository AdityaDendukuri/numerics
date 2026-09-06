/// @file container/matrix_ops.hpp
/// @brief Untagged Level-2/3 dense matrix operations: resolve through `num::accel`.
///
/// A caller who wants a specific backend calls it by name — `num::omp::matmul`,
/// `num::blas::matvec` — directly. These untagged overloads exist only for call
/// sites that do not care which backend runs.
///
/// There is one `matmul` and no tuning knobs beside it: `kernel::gemm` blocks
/// itself for the register file and the cache, so the `matmul_blocked` /
/// `matmul_register_blocked` / `matmul_simd` variants that used to sit here are
/// gone. Each was slower than the kernel now is.
#pragma once

#include "container/matrix.hpp"
#include "container/vector_ops.hpp"
#include "core/policy.hpp"
#include "kernel/kernel.hpp"
#include <algorithm>
#include <type_traits>

// `blas` and `omp` are included unconditionally. Each degrades to `num::kernel`
// internally when its library was not configured (see `blas::warn_unavailable`), so the
// namespace and its functions exist in every build — which is what lets a call site write
// `num::blas::dot(x, y)` without an `#ifdef` around it, as the docs promise. CUDA is the
// deliberate exception: it throws rather than silently running on the CPU, and its header
// needs a device toolkit, so it stays gated.
#include "blas/matrix_ops.hpp"
#include "omp/matrix_ops.hpp"
#if defined(NUMERICS_HAS_CUDA)
#include "cuda/container_ops.hpp"
#endif

namespace num::seq {

/// @brief Thin mat-aware wrappers over `num::kernel`, used when no
/// accelerator (BLAS/OMP/CUDA) was configured.
inline void matvec(const mat &A, const vec &x, vec &y) {
    kernel::matvec(y.data(), A.data(), x.data(), A.rows(), A.cols());
}

inline void matadd(real alpha, const mat &A, real beta, const mat &B, mat &C) {
    kernel::axpbyz(C.data(), A.data(), B.data(), alpha, beta, A.size());
}

inline void matmul(const mat &A, const mat &B, mat &C) {
    kernel::gemm(C.data(), A.data(), B.data(), real(1), real(0), A.rows(), B.cols(), A.cols());
}

} // namespace num::seq

namespace num {

inline void matvec(const mat &A, const vec &x, vec &y) { accel::matvec(A, x, y); }

inline void matmul(const mat &A, const mat &B, mat &C) { accel::matmul(A, B, C); }

inline void matadd(real alpha, const mat &A, real beta, const mat &B, mat &C) {
    if constexpr (requires { accel::matadd(alpha, A, beta, B, C); }) {
        accel::matadd(alpha, A, beta, B, C);
    } else {
        // Not every backend has a matadd of its own (e.g. simd/cuda don't add
        // one beyond what seq already does); fall back to the portable version.
        seq::matadd(alpha, A, beta, B, C);
    }
}


// -----------------------------------------------------------------------------
// basic_mat::apply implementation
// -----------------------------------------------------------------------------

template <std::floating_point T>
template <class X, class Y>
inline void basic_mat<T>::apply(const X &x, Y &y) const {
    if constexpr (std::is_same_v<T, real> && std::is_same_v<X, vec> && std::is_same_v<Y, vec>) {
        matvec(*this, x, y);
    } else {
        for (idx i = 0; i < rows_; ++i) {
            T sum = T(0);
            for (idx j = 0; j < cols_; ++j) {
                sum += (*this)(i, j) * x[j];
            }
            y[i] = sum;
        }
    }
}

} // namespace num
