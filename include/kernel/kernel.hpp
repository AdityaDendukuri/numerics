/// @file kernel/kernel.hpp
/// @brief Tier-0 umbrella: raw compute over pointers and callables.
///
/// SPDX-License-Identifier: MIT
/// Part of numerics, (c) 2026 Aditya Dendukuri.
/// https://github.com/AdityaDendukuri/numerics
///
/// Everything reachable from this header is templated on the scalar type, takes
/// raw pointers and callables, allocates nothing, and includes nothing outside
/// the standard library. No containers, no concepts, no algebra: the kernel
/// computes and makes no claims about what it computes on.
///
/// That is what makes this tier copyable. A consuming project can vendor these
/// files, or a single routine out of them, without adopting anything else.
#pragma once

#include "kernel/complex.hpp"
#include "kernel/dense.hpp"
#include "kernel/factor.hpp"
#include "kernel/krylov.hpp"
#include "kernel/rotations.hpp"
#include "kernel/sparse.hpp"
#include "kernel/vector.hpp"
