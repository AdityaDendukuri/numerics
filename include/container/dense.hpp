/// @file container/dense.hpp
/// @brief Rank-1 updates and triangular solves on `mat` and `vec`.
#pragma once

#include "blas/matrix_ops.hpp"
#include "container/matrix.hpp"
#include "container/vector.hpp"
#include "core/policy.hpp"
#include "core/types.hpp"
#include "kernel/kernel.hpp"
#include <algorithm>
#include <stdexcept>

namespace num::seq {

/// @brief Rank-1 update (GER): \f$A \leftarrow A + \alpha x y^T\f$.
inline void ger(real alpha, const vec<real> &x, const vec<real> &y, mat<real> &A) noexcept {
    kernel::ger(A.data(), x.data(), y.data(), alpha, x.size(), y.size());
}

/// @brief Lower triangular solve: \f$x \leftarrow L^{-1} b\f$. `x` may alias `b`.
inline void trsv_lower(const mat<real> &L, const vec<real> &b, vec<real> &x) {
    const idx n = L.rows();
    if (L.cols() != n || b.size() != n) {
        throw std::invalid_argument("trsv_lower: dimension mismatch");
    }
    if (x.size() != n) {
        x = vec<real>(n);
    }
    kernel::trsv_lower(kernel::contract::alias_safe, x.data(), L.data(), b.data(), n);
}

/// @brief Upper triangular solve: \f$x \leftarrow U^{-1} b\f$. `x` may alias `b`.
inline void trsv_upper(const mat<real> &U, const vec<real> &b, vec<real> &x) {
    const idx n = U.rows();
    if (U.cols() != n || b.size() != n) {
        throw std::invalid_argument("trsv_upper: dimension mismatch");
    }
    if (x.size() != n) {
        x = vec<real>(n);
    }
    kernel::trsv_upper(kernel::contract::alias_safe, x.data(), U.data(), b.data(), n);
}

} // namespace num::seq

#if defined(NUMERICS_HAS_OMP)
namespace num::omp {

/// @brief Rank-1 update (GER), OpenMP: \f$A \leftarrow A + \alpha x y^T\f$.
inline void ger(real alpha, const vec<real> &x, const vec<real> &y, mat<real> &A) {
    constexpr idx block_size = 64;
    const idx m = x.size();
    const idx n = y.size();
    const real *xd = x.data();
    const real *yd = y.data();
    real *ad = A.data();
#pragma omp parallel for schedule(static)
    for (idx ii = 0; ii < m; ii += block_size) {
        const idx rows = std::min(block_size, m - ii);
        kernel::ger(ad + (ii * n), n, xd + ii, yd, alpha, rows, n);
    }
}

} // namespace num::omp
#endif

namespace num {

/// @brief Rank-1 update (GER): \f$A \leftarrow A + \alpha x y^T\f$.
inline void ger(real alpha, const vec<real> &x, const vec<real> &y, mat<real> &A) {
#if defined(NUMERICS_HAS_BLAS)
    blas::ger(alpha, x, y, A);
#elif defined(NUMERICS_HAS_OMP)
    omp::ger(alpha, x, y, A);
#else
    seq::ger(alpha, x, y, A);
#endif
}

/// @brief Lower triangular solve: \f$x \leftarrow L^{-1} b\f$. `x` may alias `b`.
inline void trsv_lower(const mat<real> &L, const vec<real> &b, vec<real> &x) {
#if defined(NUMERICS_HAS_BLAS)
    blas::trsv_lower(L, b, x);
#else
    seq::trsv_lower(L, b, x);
#endif
}

/// @brief Upper triangular solve: \f$x \leftarrow U^{-1} b\f$. `x` may alias `b`.
inline void trsv_upper(const mat<real> &U, const vec<real> &b, vec<real> &x) {
#if defined(NUMERICS_HAS_BLAS)
    blas::trsv_upper(U, b, x);
#else
    seq::trsv_upper(U, b, x);
#endif
}

} // namespace num
