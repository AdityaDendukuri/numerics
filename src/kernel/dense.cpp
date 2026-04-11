/// @file kernel/dense.cpp
/// @brief Implementations for num::kernel::dense (seq_t, par_t, and no-policy ops).
///
/// ger seq_t: calls raw::ger (routes to cblas_dger when NUMERICS_HAS_BLAS).
/// ger par_t: OMP parallel-for over rows (each row is independent).
///
/// trsv: no policy; calls raw::trsv which routes to cblas_dtrsv when
/// NUMERICS_HAS_BLAS, otherwise sequential forward/back substitution.

#include "kernel/dense.hpp"
#include "kernel/raw.hpp"
#include <stdexcept>

namespace num::kernel::dense {

// ---------------------------------------------------------------------------
// ger
// ---------------------------------------------------------------------------

void ger(real alpha, const Vector& x, const Vector& y, Matrix& A,
         seq_t) noexcept {
    raw::ger(A.data(), x.data(), y.data(), alpha, x.size(), y.size());
}

void ger(real alpha, const Vector& x, const Vector& y, Matrix& A, par_t) {
#ifdef NUMERICS_HAS_OMP
    const idx   m  = x.size();
    const idx   n  = y.size();
    const real* xd = x.data();
    const real* yd = y.data();
    real*       Ad = A.data();
    #pragma omp parallel for schedule(static)
    for (idx i = 0; i < m; ++i) {
        const real axi = alpha * xd[i];
        real*      row = Ad + (i * n);
        for (idx j = 0; j < n; ++j) {
            row[j] += axi * yd[j];
        }
    }
#else
    ger(alpha, x, y, A, seq_t{});
#endif
}

// ---------------------------------------------------------------------------
// trsv_lower
// ---------------------------------------------------------------------------

void trsv_lower(const Matrix& L, const Vector& b, Vector& x) {
    const idx n = L.rows();
    if (L.cols() != n || b.size() != n) {
        throw std::invalid_argument("kernel::dense::trsv_lower: dimension mismatch");
    }
    if (x.size() != n) {
        x = Vector(n);
    }
    raw::trsv_lower(x.data(), L.data(), b.data(), n);
}

// ---------------------------------------------------------------------------
// trsv_upper
// ---------------------------------------------------------------------------

void trsv_upper(const Matrix& U, const Vector& b, Vector& x) {
    const idx n = U.rows();
    if (U.cols() != n || b.size() != n) {
        throw std::invalid_argument("kernel::dense::trsv_upper: dimension mismatch");
    }
    if (x.size() != n) {
        x = Vector(n);
    }
    raw::trsv_upper(x.data(), U.data(), b.data(), n);
}

} // namespace num::kernel::dense
