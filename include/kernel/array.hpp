/// @file kernel/array.hpp
/// @brief Elementwise vector kernels  (namespace num::kernel::array)
///
/// All operations work element-by-element over a Vector.
/// The central property is single-pass memory access: fused operations read
/// each array once, avoiding redundant loads that separate calls would incur.
///
/// Dispatched operations (seq_t / par_t overloads + default):
///   axpby(a, x, b, y)      -- y[i] = a*x[i] + b*y[i]   (fused; no BLAS eq.)
///   axpbyz(a, x, b, y, z)  -- z[i] = a*x[i] + b*y[i]   (fused; no BLAS eq.)
///
/// Always-inline template operations (policy not needed; compiler sees body):
///   map(x, f)              -- x[i] = f(x[i])
///   map(cx, f)             -- CVector variant
///   zip_map(x, y, z, f)    -- z[i] = f(x[i], y[i])
///   reduce(x, init, f)     -- left fold over x
///
/// Include kernel/kernel.hpp to get all kernel sub-modules together.
#pragma once

#include "core/types.hpp"
#include "core/vector.hpp"
#include "kernel/policy.hpp"

namespace num::kernel::array {

// ---------------------------------------------------------------------------
// axpby: y[i] = a*x[i] + b*y[i]
// ---------------------------------------------------------------------------

/// @brief Sequential: y[i] = a*x[i] + b*y[i]  (single-pass; calls raw::axpby)
void axpby(real a, const Vector& x, real b, Vector& y, seq_t) noexcept;

/// @brief Parallel: y[i] = a*x[i] + b*y[i]  (OMP parallel-for; falls back to
/// seq_t when NUMERICS_HAS_OMP is not defined)
void axpby(real a, const Vector& x, real b, Vector& y, par_t);

/// @brief Default policy (par_t if OMP available, seq_t otherwise)
inline void axpby(real a, const Vector& x, real b, Vector& y) {
    axpby(a, x, b, y, default_policy{});
}

// ---------------------------------------------------------------------------
// axpbyz: z[i] = a*x[i] + b*y[i]
// ---------------------------------------------------------------------------

/// @brief Sequential: z[i] = a*x[i] + b*y[i]  (single-pass; calls raw::axpbyz)
void axpbyz(real a, const Vector& x, real b, const Vector& y, Vector& z,
            seq_t) noexcept;

/// @brief Parallel: z[i] = a*x[i] + b*y[i]  (OMP parallel-for)
void axpbyz(real a, const Vector& x, real b, const Vector& y, Vector& z,
            par_t);

/// @brief Default policy
inline void axpbyz(real a, const Vector& x, real b, const Vector& y,
                   Vector& z) {
    axpbyz(a, x, b, y, z, default_policy{});
}

// ---------------------------------------------------------------------------
// Template kernels  (always inline; lambda is fully visible to the compiler)
// ---------------------------------------------------------------------------

/// @brief In-place elementwise map: x[i] = f(x[i])
///
/// The compiler inlines f into the loop and can vectorize if f is a simple
/// arithmetic expression. No policy needed.
///
/// @code
///   num::kernel::array::map(v, [](real x) { return x * x; });
/// @endcode
template<typename F>
void map(Vector& x, F&& f) {
    real*     d = x.data();
    const idx n = x.size();
    for (idx i = 0; i < n; ++i) {
        d[i] = f(d[i]);
    }
}

/// @brief In-place elementwise map on CVector
template<typename F>
void map(CVector& x, F&& f) {
    cplx*     d = x.data();
    const idx n = x.size();
    for (idx i = 0; i < n; ++i) {
        d[i] = f(d[i]);
    }
}

/// @brief Fused binary map: z[i] = f(x[i], y[i])
///
/// Single pass over memory; no temporary allocation.
/// x, y, z must have the same size.
///
/// @code
///   num::kernel::array::zip_map(x, y, z,
///       [](real a, real b) { return a * b; });
/// @endcode
template<typename T, typename F>
void zip_map(const BasicVector<T>& x, const BasicVector<T>& y,
             BasicVector<T>& z, F&& f) {
    const idx n = x.size();
    for (idx i = 0; i < n; ++i) {
        z[i] = f(x[i], y[i]);
    }
}

/// @brief Single-pass left fold: f(f(f(init, x[0]), x[1]), ..., x[n-1])
///
/// @code
///   real max_val = num::kernel::array::reduce(v, -1e300,
///       [](real acc, real xi) { return acc > xi ? acc : xi; });
/// @endcode
template<typename F>
[[nodiscard]] real reduce(const Vector& x, real init, F&& f) {
    const real* d   = x.data();
    const idx   n   = x.size();
    real        acc = init;
    for (idx i = 0; i < n; ++i) {
        acc = f(acc, d[i]);
    }
    return acc;
}

} // namespace num::kernel::array
