/// @file container/reduce.hpp
/// @brief scalar reduction kernels: `num::seq`/`num::omp`, and the untagged
/// `num::l1_norm`/`num::linf_norm`/`num::sum` resolving through `num::accel`.
#pragma once

#include "container/vector.hpp"
#include "container/vector_ops.hpp"
#include "core/policy.hpp"
#include "kernel/kernel.hpp"
#include "omp/parallel_ops.hpp"
#include <cmath>

namespace num::seq {

/// @brief \f$L_1\f$ norm \f$\|\mathbf{x}\|_1 = \sum_{i=0}^{n-1} |x_i|\f$.
[[nodiscard]] inline real l1_norm(const vec &x) noexcept { return kernel::l1_norm(x.data(), x.size()); }

/// @brief \f$L_\infty\f$ norm \f$\|\mathbf{x}\|_\infty = \max_i |x_i|\f$.
[[nodiscard]] inline real linf_norm(const vec &x) noexcept {
    return kernel::linf_norm(x.data(), x.size());
}

/// @brief Sum \f$\sum_{i=0}^{n-1} x_i\f$.
[[nodiscard]] inline real sum(const vec &x) noexcept { return kernel::sum(x.data(), x.size()); }

} // namespace num::seq

namespace num::omp {

[[nodiscard]] inline real l1_norm(const vec &x) {
#if defined(NUMERICS_HAS_OMP)
    const idx n = x.size();
    if (n < parallel_threshold) {
        return seq::l1_norm(x);
    }
    const real *xd = x.data();
    const idx blocks = (n + parallel_block - 1) / parallel_block;
    real s = real(0);
#pragma omp parallel for reduction(+ : s) schedule(static)
    for (idx b = 0; b < blocks; ++b) {
        const idx lo = b * parallel_block;
        s += kernel::l1_norm(xd + lo, std::min(parallel_block, n - lo));
    }
    return s;
#else
    return seq::l1_norm(x);
#endif
}

[[nodiscard]] inline real linf_norm(const vec &x) {
#if defined(NUMERICS_HAS_OMP)
    const idx n = x.size();
    if (n < parallel_threshold) {
        return seq::linf_norm(x);
    }
    const real *xd = x.data();
    const idx blocks = (n + parallel_block - 1) / parallel_block;
    real mx = real(0);
#pragma omp parallel for reduction(max : mx) schedule(static)
    for (idx b = 0; b < blocks; ++b) {
        const idx lo = b * parallel_block;
        const real v = kernel::linf_norm(xd + lo, std::min(parallel_block, n - lo));
        if (v > mx) {
            mx = v;
        }
    }
    return mx;
#else
    return seq::linf_norm(x);
#endif
}

[[nodiscard]] inline real sum(const vec &x) {
#if defined(NUMERICS_HAS_OMP)
    const idx n = x.size();
    if (n < parallel_threshold) {
        return seq::sum(x);
    }
    const real *xd = x.data();
    const idx blocks = (n + parallel_block - 1) / parallel_block;
    real s = real(0);
#pragma omp parallel for reduction(+ : s) schedule(static)
    for (idx b = 0; b < blocks; ++b) {
        const idx lo = b * parallel_block;
        s += kernel::sum(xd + lo, std::min(parallel_block, n - lo));
    }
    return s;
#else
    return seq::sum(x);
#endif
}

} // namespace num::omp

namespace num {

[[nodiscard]] inline real l1_norm(const vec &x) {
#if defined(NUMERICS_HAS_OMP)
    return omp::l1_norm(x);
#else
    return seq::l1_norm(x);
#endif
}

[[nodiscard]] inline real linf_norm(const vec &x) {
#if defined(NUMERICS_HAS_OMP)
    return omp::linf_norm(x);
#else
    return seq::linf_norm(x);
#endif
}

[[nodiscard]] inline real sum(const vec &x) {
#if defined(NUMERICS_HAS_OMP)
    return omp::sum(x);
#else
    return seq::sum(x);
#endif
}

} // namespace num
