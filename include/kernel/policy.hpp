/// @file kernel/policy.hpp
/// @brief Compile-time dispatch policy tags for the kernel module.
#pragma once

#include "core/policy.hpp" // has_omp

namespace num::kernel {

/// @brief Sequential execution policy tag.
struct seq_t {};

/// @brief Parallel execution policy tag.
struct par_t {};

inline constexpr seq_t kseq{};
inline constexpr par_t kpar{};

#if defined(NUMERICS_HAS_OMP)
using default_policy = par_t;
#else
using default_policy = seq_t;
#endif

inline constexpr default_policy kdefault{};

} // namespace num::kernel
