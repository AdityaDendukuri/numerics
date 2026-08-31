/// @file container/reduce.hpp
/// @brief Scalar reduction kernels  (namespace num::kernel::reduce)
#pragma once

#include "kernel/raw.hpp"
#include <cmath>
#include "core/types.hpp"
#include "container/vector.hpp"
#include "core/policy.hpp"

namespace num::kernel::reduce {

/// @brief Sequential \f$L_1\f$ norm \f$\|\mathbf{x}\|_1 = \sum_{i=0}^{n-1} |x_i|\f$.
[[nodiscard]] real l1_norm(const Vector &x, seq_t) noexcept;

/// @brief Parallel \f$L_1\f$ norm \f$\|\mathbf{x}\|_1 = \sum_{i=0}^{n-1} |x_i|\f$.
[[nodiscard]] real l1_norm(const Vector &x, par_t);

[[nodiscard]] inline real l1_norm(const Vector &x) {
    return l1_norm(x, default_policy{});
}

/// @brief Sequential \f$L_\infty\f$ norm \f$\|\mathbf{x}\|_\infty = \max_i |x_i|\f$.
[[nodiscard]] real linf_norm(const Vector &x, seq_t) noexcept;

/// @brief Parallel \f$L_\infty\f$ norm \f$\|\mathbf{x}\|_\infty = \max_i |x_i|\f$.
[[nodiscard]] real linf_norm(const Vector &x, par_t);

[[nodiscard]] inline real linf_norm(const Vector &x) {
    return linf_norm(x, default_policy{});
}

/// @brief Sequential sum \f$\sum_{i=0}^{n-1} x_i\f$.
[[nodiscard]] real sum(const Vector &x, seq_t) noexcept;

/// @brief Parallel sum \f$\sum_{i=0}^{n-1} x_i\f$.
[[nodiscard]] real sum(const Vector &x, par_t);

[[nodiscard]] inline real sum(const Vector &x) {
    return sum(x, default_policy{});
}



inline real l1_norm(const Vector &x, seq_t) noexcept {
    return raw::l1_norm(x.data(), x.size());
}

inline real l1_norm(const Vector &x, par_t) {
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

inline real linf_norm(const Vector &x, seq_t) noexcept {
    return raw::linf_norm(x.data(), x.size());
}

inline real linf_norm(const Vector &x, par_t) {
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

inline real sum(const Vector &x, seq_t) noexcept {
    return raw::sum(x.data(), x.size());
}

inline real sum(const Vector &x, par_t) {
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
