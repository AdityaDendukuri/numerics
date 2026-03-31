/// @file core/backends/blas/vector.cpp
/// @brief BLAS backend  -- cblas level-1 vector operations

#include "core/vector.hpp"
#include "../seq/impl.hpp"
#include <cmath>
#include <cstdio>

#ifdef NUMERICS_HAS_BLAS
    #include <cblas.h>
#endif

namespace {
void warn_blas_unavailable() {
#ifndef NUMERICS_HAS_BLAS
    static bool warned = false;
    if (!warned) {
        warned = true;
        std::fprintf(stderr,
                     "[numerics] WARNING: Backend::blas requested but BLAS was "
                     "not found at "
                     "configure time.\n"
                     "           Falling back to Backend::seq.\n"
                     "           Install OpenBLAS and reconfigure: "
                     "apt install libopenblas-dev\n");
    }
#endif
}
} // namespace

namespace num::backends::blas {

void scale(Vector& v, real alpha) {
    warn_blas_unavailable();
#ifdef NUMERICS_HAS_BLAS
    cblas_dscal(static_cast<int>(v.size()), alpha, v.data(), 1);
#else
    num::backends::seq::scale(v, alpha);
#endif
}

void axpy(real alpha, const Vector& x, Vector& y) {
    warn_blas_unavailable();
#ifdef NUMERICS_HAS_BLAS
    cblas_daxpy(static_cast<int>(x.size()), alpha, x.data(), 1, y.data(), 1);
#else
    num::backends::seq::axpy(alpha, x, y);
#endif
}

real dot(const Vector& x, const Vector& y) {
    warn_blas_unavailable();
#ifdef NUMERICS_HAS_BLAS
    return cblas_ddot(static_cast<int>(x.size()), x.data(), 1, y.data(), 1);
#else
    return num::backends::seq::dot(x, y);
#endif
}

real norm(const Vector& x) {
    warn_blas_unavailable();
#ifdef NUMERICS_HAS_BLAS
    return cblas_dnrm2(static_cast<int>(x.size()), x.data(), 1);
#else
    return num::backends::seq::norm(x);
#endif
}

} // namespace num::backends::blas
