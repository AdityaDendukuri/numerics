/// @file container/matrix_ops.hpp
/// @brief Untagged Level-2/3 dense matrix operations: resolve through `num::accel`.
///
/// Call a backend by name, such as `num::omp::matmul`, to pick one. `kernel::gemm` blocks
/// itself, so `matmul` has no tuning variants.
#pragma once

#include "container/concepts.hpp"
#include "container/matrix.hpp"
#include "container/vector_ops.hpp"
#include "core/policy.hpp"
#include "kernel/kernel.hpp"
#include <algorithm>
#include <type_traits>

// `blas` and `omp` are always included and fall back to `num::kernel` when unconfigured, so
// `num::blas::dot(x, y)` needs no `#ifdef`. CUDA throws instead and needs a toolkit, so it
// stays gated.
#include "blas/matrix_ops.hpp"
#include "omp/matrix_ops.hpp"
#if defined(NUMERICS_HAS_CUDA)
#include "cuda/container_ops.hpp"
#endif

namespace num::seq {

/// @brief Thin mat-aware wrappers over `num::kernel`, used when no
/// accelerator (BLAS/OMP/CUDA) was configured.
inline void matvec(const mat<real> &A, const vec<real> &x, vec<real> &y) {
    kernel::matvec(y.data(), A.data(), x.data(), A.rows(), A.cols());
}

inline void matadd(real alpha, const mat<real> &A, real beta, const mat<real> &B, mat<real> &C) {
    kernel::axpbyz(C.data(), A.data(), B.data(), alpha, beta, A.size());
}

inline void matmul(const mat<real> &A, const mat<real> &B, mat<real> &C) {
    kernel::gemm(C.data(), A.data(), B.data(), real(1), real(0), A.rows(), B.cols(), A.cols());
}

} // namespace num::seq

namespace num {

/// @brief \f$y \leftarrow Ax\f$ for any row-major dense matrix.
///
/// Constrained on @ref num::repr::dense_row_major, so a foreign matrix with `data()`, `rows()`
/// and `cols()` works. `num::mat<real>` takes the configured backend, anything else the kernel. The
/// CSR overload in `linear/sparse/sparse.hpp` is selected by the disjoint @ref num::repr::csr.
template <class M>
requires repr::dense_row_major<M>
inline void matvec(const M &A, const vec<real> &x, vec<real> &y) {
    if constexpr (requires { accel::matvec(A, x, y); }) {
        accel::matvec(A, x, y);
    } else {
        kernel::matvec(y.data(), A.data(), x.data(), A.rows(), A.cols());
    }
}

inline void matmul(const mat<real> &A, const mat<real> &B, mat<real> &C) { accel::matmul(A, B, C); }

/// @brief \f$C \leftarrow \alpha A + \beta B\f$.
inline void matadd(real alpha, const mat<real> &A, real beta, const mat<real> &B, mat<real> &C) {
    if constexpr (requires { accel::matadd(alpha, A, beta, B, C); }) {
        accel::matadd(alpha, A, beta, B, C);
    } else {
        // Not every backend has a matadd of its own (e.g. simd/cuda don't add
        // one beyond what seq already does); fall back to the portable version.
        seq::matadd(alpha, A, beta, B, C);
    }
}

// mat::apply implementation

template <std::floating_point T>
template <class X, class Y>
inline void mat<T>::apply(const X &x, Y &y) const {
    if constexpr (std::is_same_v<T, real> && std::is_same_v<X, vec<real>> && std::is_same_v<Y, vec<real>>) {
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
