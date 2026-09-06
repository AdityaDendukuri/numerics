/// @file kernel/debug.hpp
/// @brief Opt-in stream formatting for kernel result types.
///
/// SPDX-License-Identifier: MIT
/// Part of numerics, (c) 2026 Aditya Dendukuri.
/// https://github.com/AdityaDendukuri/numerics
///
/// Kept out of `kernel/krylov.hpp` (and out of the `kernel/kernel.hpp` umbrella)
/// so the compute headers never pull `<ostream>`: on libc++ that costs ~38k
/// preprocessed lines against a 54k baseline, and a freestanding or embedded
/// target may have no iostreams at all. Nothing in the compute path needs it —
/// include this header only where you actually want to print a result.
///
/// The operator is declared in `num::kernel` so ADL finds it for
/// `std::cout << result` without a using-declaration.
#pragma once

#include "kernel/krylov.hpp"
#include <concepts>
#include <ostream>

namespace num::kernel {

template <std::floating_point T>
inline std::ostream &operator<<(std::ostream &os, const krylov_result<T> &r) {
    os << "krylov_result{ converged: " << (r.converged ? "true" : "false")
       << ", iterations: " << r.iterations << ", residual: " << r.residual << " }";
    return os;
}

} // namespace num::kernel
