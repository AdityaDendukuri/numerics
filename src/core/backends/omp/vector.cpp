/// @file core/backends/omp/vector.cpp
/// @brief OpenMP backend  -- parallelised vector operations

#include "core/vector.hpp"
#include "../seq/impl.hpp"

namespace num::backends::omp {

void scale(Vector& v, real alpha) {
#ifdef NUMERICS_HAS_OMP
    const idx n = v.size();
#   pragma omp parallel for schedule(static)
    for (idx i = 0; i < n; ++i) v[i] *= alpha;
#else
    num::backends::seq::scale(v, alpha);
#endif
}

void axpy(real alpha, const Vector& x, Vector& y) {
#ifdef NUMERICS_HAS_OMP
    const idx n = x.size();
#   pragma omp parallel for schedule(static)
    for (idx i = 0; i < n; ++i) y[i] += alpha * x[i];
#else
    num::backends::seq::axpy(alpha, x, y);
#endif
}

real dot(const Vector& x, const Vector& y) {
#ifdef NUMERICS_HAS_OMP
    real sum = 0;
    const idx n = x.size();
#   pragma omp parallel for reduction(+:sum) schedule(static)
    for (idx i = 0; i < n; ++i) sum += x[i] * y[i];
    return sum;
#else
    return num::backends::seq::dot(x, y);
#endif
}

} // namespace num::backends::omp
