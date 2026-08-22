/// @file kernel/array.hpp
/// @brief Elementwise vector kernels  (namespace num::kernel::array)
#pragma once

#include "core/types.hpp"
#include "core/vector.hpp"
#include "kernel/policy.hpp"

namespace num::kernel::array {

/// @brief Sequential: y[i] = a*x[i] + b*y[i]  (single-pass; calls raw::axpby)
void axpby(real a, const Vector &x, real b, Vector &y, seq_t) noexcept;

/// @brief Parallel: y[i] = a*x[i] + b*y[i].
void axpby(real a, const Vector &x, real b, Vector &y, par_t);

inline void axpby(real a, const Vector &x, real b, Vector &y) {
    axpby(a, x, b, y, default_policy{});
}

/// @brief Sequential: z[i] = a*x[i] + b*y[i].
void axpbyz(real a, const Vector &x, real b, const Vector &y, Vector &z, seq_t) noexcept;

/// @brief Parallel: z[i] = a*x[i] + b*y[i].
void axpbyz(real a, const Vector &x, real b, const Vector &y, Vector &z, par_t);

inline void axpbyz(real a, const Vector &x, real b, const Vector &y, Vector &z) {
    axpbyz(a, x, b, y, z, default_policy{});
}

/// @brief In-place elementwise map: x[i] = f(x[i])
template <typename T, typename F>
void map(BasicVector<T> &x, F &&f) {
    T *d = x.data();
    const idx n = x.size();
    for (idx i = 0; i < n; ++i) {
        d[i] = f(d[i]);
    }
}

/// @brief Fused binary map: z[i] = f(x[i], y[i])
template <typename T, typename F>
void zip_map(const BasicVector<T> &x, const BasicVector<T> &y, BasicVector<T> &z, F &&f) {
    const idx n = x.size();
    for (idx i = 0; i < n; ++i) {
        z[i] = f(x[i], y[i]);
    }
}

/// @brief Single-pass left fold: f(f(f(init, x[0]), x[1]), ..., x[n-1])
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

} // namespace num::kernel::array
