/// @file kernel/reduce.cpp
/// @brief Implementations for num::kernel::reduce (seq_t and par_t overloads).

#include "kernel/reduce.hpp"
#include "kernel/raw.hpp"
#include <cmath>

namespace num::kernel::reduce {

real l1_norm(const Vector &x, seq_t) noexcept {
    return raw::l1_norm(x.data(), x.size());
}

real l1_norm(const Vector &x, par_t) {
#ifdef NUMERICS_HAS_OMP
    const idx n = x.size();
    const real *xd = x.data();
    real s = real(0);
#pragma omp parallel for reduction(+ : s) schedule(static)
    for (idx i = 0; i < n; ++i) {
        s += std::abs(xd[i]);
    }
    return s;
#else
    return l1_norm(x, seq_t{});
#endif
}

real linf_norm(const Vector &x, seq_t) noexcept {
    return raw::linf_norm(x.data(), x.size());
}

real linf_norm(const Vector &x, par_t) {
#ifdef NUMERICS_HAS_OMP
    const idx n = x.size();
    const real *xd = x.data();
    real mx = real(0);
#pragma omp parallel for reduction(max : mx) schedule(static)
    for (idx i = 0; i < n; ++i) {
        const real v = std::abs(xd[i]);
        if (v > mx) {
            mx = v;
        }
    }
    return mx;
#else
    return linf_norm(x, seq_t{});
#endif
}

real sum(const Vector &x, seq_t) noexcept {
    return raw::sum(x.data(), x.size());
}

real sum(const Vector &x, par_t) {
#ifdef NUMERICS_HAS_OMP
    const idx n = x.size();
    const real *xd = x.data();
    real s = real(0);
#pragma omp parallel for reduction(+ : s) schedule(static)
    for (idx i = 0; i < n; ++i) {
        s += xd[i];
    }
    return s;
#else
    return sum(x, seq_t{});
#endif
}

} // namespace num::kernel::reduce
