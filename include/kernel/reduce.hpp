/// @file kernel/reduce.hpp
/// @brief Scalar reduction kernels  (namespace num::kernel::reduce)
///
/// These operations collapse a Vector to a single scalar via a single memory
/// pass. Unlike dot/norm (which live in num:: with full Backend dispatch),
/// these are new reductions that have no existing parallel paths:
///
///   l1_norm(x)   -- sum |x[i]|    (BLAS cblas_dasum via raw:: on seq_t path)
///   linf_norm(x) -- max |x[i]|    (BLAS cblas_idamax via raw:: on seq_t path)
///   sum(x)       -- sum x[i]      (no BLAS equivalent; seq loop or OMP)
///
/// Each has seq_t / par_t overloads and a default that selects par_t when
/// NUMERICS_HAS_OMP is defined at configure time.
///
/// Include kernel/kernel.hpp to get all kernel sub-modules together.
#pragma once

#include "core/types.hpp"
#include "core/vector.hpp"
#include "kernel/policy.hpp"

namespace num::kernel::reduce {

// ---------------------------------------------------------------------------
// l1_norm: sum |x[i]|
// ---------------------------------------------------------------------------

/// @brief Sequential: calls raw::l1_norm (routes to cblas_dasum when BLAS
/// available; otherwise auto-vectorizable seq loop).
[[nodiscard]] real l1_norm(const Vector& x, seq_t) noexcept;

/// @brief Parallel: OMP reduction(+) over abs values; falls back to seq_t
/// when NUMERICS_HAS_OMP is not defined.
[[nodiscard]] real l1_norm(const Vector& x, par_t);

/// @brief Default policy
[[nodiscard]] inline real l1_norm(const Vector& x) {
    return l1_norm(x, default_policy{});
}

// ---------------------------------------------------------------------------
// linf_norm: max |x[i]|
// ---------------------------------------------------------------------------

/// @brief Sequential: calls raw::linf_norm (routes to cblas_idamax when BLAS
/// available; otherwise plain max loop).
[[nodiscard]] real linf_norm(const Vector& x, seq_t) noexcept;

/// @brief Parallel: OMP reduction(max) over abs values; falls back to seq_t
/// when NUMERICS_HAS_OMP is not defined.
[[nodiscard]] real linf_norm(const Vector& x, par_t);

/// @brief Default policy
[[nodiscard]] inline real linf_norm(const Vector& x) {
    return linf_norm(x, default_policy{});
}

// ---------------------------------------------------------------------------
// sum: sum x[i]
// ---------------------------------------------------------------------------

/// @brief Sequential: auto-vectorizable summation loop (no BLAS equivalent).
[[nodiscard]] real sum(const Vector& x, seq_t) noexcept;

/// @brief Parallel: OMP reduction(+) over x[i]; falls back to seq_t
/// when NUMERICS_HAS_OMP is not defined.
[[nodiscard]] real sum(const Vector& x, par_t);

/// @brief Default policy
[[nodiscard]] inline real sum(const Vector& x) {
    return sum(x, default_policy{});
}

} // namespace num::kernel::reduce
