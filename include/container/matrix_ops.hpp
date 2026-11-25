/// @file container/matrix_ops.hpp
/// @brief Dense matrix operations through the default backend.
#pragma once

#include "container/concepts.hpp"
#include "container/matrix.hpp"
#include "container/vector_ops.hpp"
#include "core/policy.hpp"
#include "kernel/kernel.hpp"
#include <algorithm>
#include <stdexcept>
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

/// @brief Matrix-vector product, sequential: \f$y \leftarrow A x\f$.
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

/// @brief Matrix-vector product for any row-major matrix: \f$y \leftarrow A x\f$.
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

/// @brief Matrix-matrix product (GEMM): \f$C \leftarrow \alpha\, op(A)\, op(B) + \beta C\f$.
/// \f$op\f$ transposes when its flag is set.
inline void gemm(real alpha, const mat<real> &A, bool transA, const mat<real> &B, bool transB, real beta,
                 mat<real> &C) {
    blas::gemm(alpha, A, transA, B, transB, beta, C);
}

/// @brief Matrix-vector product (GEMV): \f$y \leftarrow \alpha\, op(A_r)\, x + \beta y\f$.
/// \f$A_r\f$ is the leading `rows` rows of A; `y` must already be sized.
inline void gemv(real alpha, const mat<real> &A, bool transA, const vec<real> &x, real beta, vec<real> &y,
                 idx rows = static_cast<idx>(-1)) {
    const idx r = std::min(rows, A.rows());
    const idx m = transA ? A.cols() : r, k = transA ? r : A.cols();
    if (x.size() != k || y.size() != m) {
        throw std::invalid_argument("gemv: dimensions do not agree");
    }
    blas::gemm(alpha, A.data(), A.cols(), transA, x.data(), 1, false, beta, y.data(), 1, m, 1, k);
}

/// @brief Matrix linear combination: \f$C \leftarrow \alpha A + \beta B\f$.
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
