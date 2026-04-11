/// @file kernel/policy.hpp
/// @brief Compile-time dispatch policy tags for the kernel module.
///
/// The kernel layer has one relevant dimension of variation: parallelism.
/// BLAS dispatch is already handled inside raw:: for ops that have cblas
/// equivalents. Vectorization (SIMD) is handled by the compiler or by raw::.
/// The only thing the caller needs to choose is: run sequentially or in
/// parallel (OMP).
///
/// Two tags:
///   seq_t  -- always sequential; safe to call from inside an OMP region
///   par_t  -- OpenMP parallel; falls through to seq_t if OMP not available
///
/// Default policy (kdefault / default_policy):
///   par_t  when NUMERICS_HAS_OMP is defined
///   seq_t  otherwise
///
/// Usage:
///   num::kernel::array::axpby(a, x, b, y);              // default policy
///   num::kernel::array::axpby(a, x, b, y, num::kernel::kseq);  // force seq
///   num::kernel::array::axpby(a, x, b, y, num::kernel::kpar);  // force par
///
/// For templates that cascade policy through multiple kernel calls:
///   template<typename Policy = num::kernel::default_policy>
///   void my_kernel(..., Policy p = {}) {
///       num::kernel::reduce::l1_norm(x, p);
///       num::kernel::array::axpby(a, x, b, y, p);
///   }
#pragma once

#include "core/policy.hpp"   // has_omp

namespace num::kernel {

/// @brief Sequential execution policy tag.
/// Guarantees no OMP parallel regions; safe to call inside an existing
/// parallel region without causing nested-parallelism overhead.
struct seq_t {};

/// @brief Parallel execution policy tag.
/// Activates OMP parallel-for / reduction constructs when NUMERICS_HAS_OMP.
/// Falls through to seq_t behaviour when OMP is not available.
struct par_t {};

inline constexpr seq_t kseq{};
inline constexpr par_t kpar{};

/// @brief Default policy: par_t if OMP is available, seq_t otherwise.
/// Selected at configure time; zero runtime overhead.
#if defined(NUMERICS_HAS_OMP)
using default_policy = par_t;
#else
using default_policy = seq_t;
#endif

inline constexpr default_policy kdefault{};

} // namespace num::kernel
