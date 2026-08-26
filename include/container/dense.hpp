/// @file container/dense.hpp
/// @brief Dense matrix inner kernels  (namespace num::kernel::dense)
#pragma once

#include "container/matrix.hpp"
#include "container/vector.hpp"
#include "core/policy.hpp"
#include "core/types.hpp"
#include "kernel/raw.hpp"
#include <stdexcept>

namespace num::kernel::dense {

/// @brief Sequential rank-1 matrix update \f$A \leftarrow A + \alpha \mathbf{x} \mathbf{y}^T\f$.
void ger(real alpha, const Vector &x, const Vector &y, Matrix &A, seq_t) noexcept;

/// @brief Parallel rank-1 matrix update \f$A \leftarrow A + \alpha \mathbf{x} \mathbf{y}^T\f$.
void ger(real alpha, const Vector &x, const Vector &y, Matrix &A, par_t);

inline void ger(real alpha, const Vector &x, const Vector &y, Matrix &A) {
    ger(alpha, x, y, A, default_policy{});
}

/// @brief Forward substitution solving lower triangular system \f$L \mathbf{x} = \mathbf{b}\f$.
void trsv_lower(const Matrix &L, const Vector &b, Vector &x);

/// @brief Back substitution solving upper triangular system \f$U \mathbf{x} = \mathbf{b}\f$.
void trsv_upper(const Matrix &U, const Vector &b, Vector &x);

inline void ger(real alpha, const Vector &x, const Vector &y, Matrix &A, seq_t) noexcept {
    raw::ger(A.data(), x.data(), y.data(), alpha, x.size(), y.size());
}

inline void ger(real alpha, const Vector &x, const Vector &y, Matrix &A, par_t) {
#ifdef NUMERICS_HAS_OMP
    const idx m = x.size();
    const idx n = y.size();
    const real *xd = x.data();
    const real *yd = y.data();
    real *ad = A.data();
#pragma omp parallel for schedule(static)
    for (idx i = 0; i < m; ++i) {
        const real axi = alpha * xd[i];
        real *row = ad + (i * n);
        for (idx j = 0; j < n; ++j) {
            row[j] += axi * yd[j];
        }
    }
#else
    ger(alpha, x, y, A, seq_t{});
#endif
}

inline void trsv_lower(const Matrix &L, const Vector &b, Vector &x) {
    const idx n = L.rows();
    if (L.cols() != n || b.size() != n) {
        throw std::invalid_argument("kernel::dense::trsv_lower: dimension mismatch");
    }
    if (x.size() != n) {
        x = Vector(n);
    }
    raw::trsv_lower(raw::contract::alias_safe, x.data(), L.data(), b.data(), n);
}

inline void trsv_upper(const Matrix &U, const Vector &b, Vector &x) {
    const idx n = U.rows();
    if (U.cols() != n || b.size() != n) {
        throw std::invalid_argument("kernel::dense::trsv_upper: dimension mismatch");
    }
    if (x.size() != n) {
        x = Vector(n);
    }
    raw::trsv_upper(raw::contract::alias_safe, x.data(), U.data(), b.data(), n);
}

} // namespace num::kernel::dense
