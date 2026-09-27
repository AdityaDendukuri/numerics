/// @file kernel/kernel.hpp
/// @brief Tier-0 umbrella: raw compute over pointers and callables.
///
/// SPDX-License-Identifier: MIT
/// Part of numerics, (c) 2026 Aditya Dendukuri.
/// https://github.com/AdityaDendukuri/numerics
///
/// Everything here is templated on the scalar, takes raw pointers and callables, allocates
/// nothing and needs only the standard library, so the files can be vendored alone.
#pragma once

#include "kernel/complex.hpp"
#include "kernel/dense.hpp"
#include "kernel/factor.hpp"
#include "kernel/krylov.hpp"
#include "kernel/rotations.hpp"
#include "kernel/sparse.hpp"
#include "kernel/vector.hpp"
