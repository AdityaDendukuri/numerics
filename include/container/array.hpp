/// @file container/array.hpp
/// @brief Elementwise vector kernels  (namespace num::kernel::array)
#pragma once

#include "kernel/raw.hpp"
#include "core/types.hpp"
#include "container/vector.hpp"
#include "core/policy.hpp"

namespace num::kernel::array {

/// @brief Computes scaled vector update \f$y_i \leftarrow a x_i + b y_i\f$ sequentially in a single memory pass.
void axpby(real a, const Vector &x, real b, Vector &y, seq_t) noexcept;

/// @brief Computes scaled vector update \f$y_i \leftarrow a x_i + b y_i\f$ in parallel.
void axpby(real a, const Vector &x, real b, Vector &y, par_t);

inline void axpby(real a, const Vector &x, real b, Vector &y) {
    axpby(a, x, b, y, default_policy{});
}

/// @brief Computes linear combination \f$z_i \leftarrow a x_i + b y_i\f$ sequentially.
void axpbyz(real a, const Vector &x, real b, const Vector &y, Vector &z, seq_t) noexcept;

/// @brief Computes linear combination \f$z_i \leftarrow a x_i + b y_i\f$ in parallel.
void axpbyz(real a, const Vector &x, real b, const Vector &y, Vector &z, par_t);

inline void axpbyz(real a, const Vector &x, real b, const Vector &y, Vector &z) {
    axpbyz(a, x, b, y, z, default_policy{});
}

/// @brief In-place elementwise transformation \f$x_i \leftarrow f(x_i)\f$.
template <typename T, typename F>
void map(BasicVector<T> &x, F &&f) {
    T *d = x.data();
    const idx n = x.size();
    for (idx i = 0; i < n; ++i) {
        d[i] = f(d[i]);
    }
}

/// @brief Fused binary elementwise transformation \f$z_i \leftarrow f(x_i, y_i)\f$.
template <typename T, typename F>
void zip_map(const BasicVector<T> &x, const BasicVector<T> &y, BasicVector<T> &z, F &&f) {
    const idx n = x.size();
    for (idx i = 0; i < n; ++i) {
        z[i] = f(x[i], y[i]);
    }
}

/// @brief Single-pass left fold: \f$f(\dots f(f(\text{init}, x_0), x_1), \dots, x_{n-1})\f$.
template <typename T, typename F>
[[nodiscard]] T reduce(const BasicVector<T> &x, T init, F &&f) {
    const T *d = x.data();
    const idx n = x.size();
    T acc = init;
    for (idx i = 0; i < n; ++i) {
        acc = f(acc, d[i]);
    }
    return acc;
}



inline void axpby(real a, const Vector &x, real b, Vector &y, seq_t) noexcept {
    raw::axpby(y.data(), x.data(), a, b, x.size());
}

inline void axpby(real a, const Vector &x, real b, Vector &y, par_t) {
#ifdef NUMERICS_HAS_OMP
    const idx n = x.size();
    const real *xd = x.data();
    real *yd = y.data();
#pragma omp parallel for schedule(static)
    for (idx i = 0; i < n; ++i) {
        yd[i] = (a * xd[i]) + (b * yd[i]);
    }
#else
    axpby(a, x, b, y, seq_t{});
#endif
}

inline void axpbyz(real a, const Vector &x, real b, const Vector &y, Vector &z, seq_t) noexcept {
    raw::axpbyz(z.data(), x.data(), y.data(), a, b, x.size());
}

inline void axpbyz(real a, const Vector &x, real b, const Vector &y, Vector &z, par_t) {
#ifdef NUMERICS_HAS_OMP
    const idx n = x.size();
    const real *xd = x.data();
    const real *yd = y.data();
    real *zd = z.data();
#pragma omp parallel for schedule(static)
    for (idx i = 0; i < n; ++i) {
        zd[i] = (a * xd[i]) + (b * yd[i]);
    }
#else
    axpbyz(a, x, b, y, z, seq_t{});
#endif
}

} // namespace num::kernel::array
