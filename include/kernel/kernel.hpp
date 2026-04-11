/// @file kernel/kernel.hpp
/// @brief Master include for the numerics kernel module.
///
/// The kernel module is the performance substrate that the rest of numerics
/// sits on top of. It is organized in tiers and sub-namespaces:
///
///   Tier 1 -- raw.hpp  (namespace num::kernel::raw)
///     Raw-pointer, always-inline inner loops with __restrict__ and ivdep.
///     BLAS dispatch built in where cblas equivalents exist. No OMP.
///     Use when writing a fused custom kernel that composes multiple
///     operations into a single memory pass.
///
///   Tier 2 -- policy.hpp + array.hpp + reduce.hpp + dense.hpp
///
///     policy.hpp  (num::kernel)
///       seq_t, par_t, default_policy, kseq, kpar, kdefault
///
///     array.hpp   (num::kernel::array)
///       axpby, axpbyz           -- fused elementwise, dispatched
///       map, zip_map, reduce    -- template, always-inline
///
///     reduce.hpp  (num::kernel::reduce)
///       l1_norm, linf_norm, sum -- scalar reductions, dispatched
///
///     dense.hpp   (num::kernel::dense)
///       ger                     -- rank-1 update, dispatched
///       trsv_lower, trsv_upper  -- triangular solves (serial; BLAS via raw::)
///
/// Dispatch uses compile-time policy tags (seq_t / par_t) rather than a
/// runtime enum. The default (kdefault / default_policy) is par_t when
/// NUMERICS_HAS_OMP is defined, seq_t otherwise. Zero runtime overhead.
///
/// Quick reference:
///   num::kernel::raw::axpy(y, x, alpha, n)          -- raw pointer
///   num::kernel::array::axpby(a, x, b, y)           -- type-safe, default policy
///   num::kernel::array::axpby(a, x, b, y, kseq)     -- force sequential
///   num::kernel::reduce::l1_norm(x, kpar)            -- force parallel
///   num::kernel::dense::trsv_lower(L, b, x)          -- triangular solve
#pragma once

#include "kernel/raw.hpp"
#include "kernel/policy.hpp"
#include "kernel/array.hpp"
#include "kernel/reduce.hpp"
#include "kernel/dense.hpp"
#include "kernel/subspace.hpp"
