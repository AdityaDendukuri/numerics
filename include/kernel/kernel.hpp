/// @file kernel/kernel.hpp
/// @brief The BLAS layer over raw pointers.
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
#include "kernel/sparse.hpp"
#include "kernel/vector.hpp"
