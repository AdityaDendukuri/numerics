/// @file kernel/array.cpp
/// @brief Implementations for num::kernel::array (seq_t and par_t overloads).
///
/// seq_t paths call raw:: directly (which handles BLAS dispatch internally
/// for ops that have cblas equivalents, and uses auto-vectorizable loops
/// otherwise).
///
/// par_t paths use OpenMP parallel-for. When NUMERICS_HAS_OMP is not defined
/// par_t falls through to the seq_t implementation.

#include "kernel/array.hpp"
#include "kernel/raw.hpp"

namespace num::kernel::array {

// ---------------------------------------------------------------------------
// axpby
// ---------------------------------------------------------------------------

void axpby(real a, const Vector& x, real b, Vector& y, seq_t) noexcept {
    raw::axpby(y.data(), x.data(), a, b, x.size());
}

void axpby(real a, const Vector& x, real b, Vector& y, par_t) {
#ifdef NUMERICS_HAS_OMP
    const idx   n  = x.size();
    const real* xd = x.data();
    real*       yd = y.data();
    #pragma omp parallel for schedule(static)
    for (idx i = 0; i < n; ++i) {
        yd[i] = (a * xd[i]) + (b * yd[i]);
    }
#else
    axpby(a, x, b, y, seq_t{});
#endif
}

// ---------------------------------------------------------------------------
// axpbyz
// ---------------------------------------------------------------------------

void axpbyz(real a, const Vector& x, real b, const Vector& y, Vector& z,
            seq_t) noexcept {
    raw::axpbyz(z.data(), x.data(), y.data(), a, b, x.size());
}

void axpbyz(real a, const Vector& x, real b, const Vector& y, Vector& z,
            par_t) {
#ifdef NUMERICS_HAS_OMP
    const idx   n  = x.size();
    const real* xd = x.data();
    const real* yd = y.data();
    real*       zd = z.data();
    #pragma omp parallel for schedule(static)
    for (idx i = 0; i < n; ++i) {
        zd[i] = (a * xd[i]) + (b * yd[i]);
    }
#else
    axpbyz(a, x, b, y, z, seq_t{});
#endif
}

} // namespace num::kernel::array
