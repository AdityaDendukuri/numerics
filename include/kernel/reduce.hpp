/// @file kernel/reduce.hpp
/// @brief Scalar reduction kernels  (namespace num::kernel::reduce)
#pragma once

#include "core/types.hpp"
#include "core/vector.hpp"
#include "kernel/policy.hpp"

namespace num::kernel::reduce {

/// @brief Sequential l1 norm.
[[nodiscard]] real l1_norm(const Vector &x, seq_t) noexcept;

/// @brief Parallel l1 norm.
[[nodiscard]] real l1_norm(const Vector &x, par_t);

[[nodiscard]] inline real l1_norm(const Vector &x) {
    return l1_norm(x, default_policy{});
}

/// @brief Sequential infinity norm.
[[nodiscard]] real linf_norm(const Vector &x, seq_t) noexcept;

/// @brief Parallel infinity norm.
[[nodiscard]] real linf_norm(const Vector &x, par_t);

[[nodiscard]] inline real linf_norm(const Vector &x) {
    return linf_norm(x, default_policy{});
}

/// @brief Sequential sum.
[[nodiscard]] real sum(const Vector &x, seq_t) noexcept;

/// @brief Parallel sum.
[[nodiscard]] real sum(const Vector &x, par_t);

[[nodiscard]] inline real sum(const Vector &x) {
    return sum(x, default_policy{});
}

} // namespace num::kernel::reduce
