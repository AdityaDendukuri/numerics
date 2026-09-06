/// @file container/dense.hpp
/// @brief Dense matrix inner kernels: `num::seq::ger`/`num::omp::ger`, and the
/// untagged `num::ger`/`num::trsv_lower`/`num::trsv_upper`.
#pragma once

#include "container/matrix.hpp"
#include "container/vector.hpp"
#include "core/policy.hpp"
#include "core/types.hpp"
#include "kernel/kernel.hpp"
#include <algorithm>
#include <stdexcept>

namespace num::seq {

/// @brief Rank-1 matrix update \f$A \leftarrow A + \alpha \mathbf{x} \mathbf{y}^T\f$.
inline void ger(real alpha, const vec &x, const vec &y, mat &A) noexcept {
    kernel::ger(A.data(), x.data(), y.data(), alpha, x.size(), y.size());
}

} // namespace num::seq

#if defined(NUMERICS_HAS_OMP)
namespace num::omp {

/// @brief Rank-1 matrix update \f$A \leftarrow A + \alpha \mathbf{x} \mathbf{y}^T\f$.
///
/// Row-tiled rather than routed through `dispatch::parallel_apply`: see
/// `omp/matrix_ops.hpp` for why row/matrix operations use their own tiling
/// instead of the vector-element-tuned block helper.
inline void ger(real alpha, const vec &x, const vec &y, mat &A) {
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

inline void ger(real alpha, const vec &x, const vec &y, mat &A) {
#if defined(NUMERICS_HAS_OMP)
    omp::ger(alpha, x, y, A);
#else
    seq::ger(alpha, x, y, A);
#endif
}

/// @brief Forward substitution solving lower triangular system \f$L \mathbf{x} = \mathbf{b}\f$.
inline void trsv_lower(const mat &L, const vec &b, vec &x) {
    const idx n = L.rows();
    if (L.cols() != n || b.size() != n) {
        throw std::invalid_argument("trsv_lower: dimension mismatch");
    }
    if (x.size() != n) {
        x = vec(n);
    }
    kernel::trsv_lower(kernel::contract::alias_safe, x.data(), L.data(), b.data(), n);
}

/// @brief Back substitution solving upper triangular system \f$U \mathbf{x} = \mathbf{b}\f$.
inline void trsv_upper(const mat &U, const vec &b, vec &x) {
    const idx n = U.rows();
    if (U.cols() != n || b.size() != n) {
        throw std::invalid_argument("trsv_upper: dimension mismatch");
    }
    if (x.size() != n) {
        x = vec(n);
    }
    kernel::trsv_upper(kernel::contract::alias_safe, x.data(), U.data(), b.data(), n);
}

} // namespace num
