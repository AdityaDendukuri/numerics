/// @file kernel/debug.hpp
/// @brief Opt-in stream formatting for kernel result types.
///
/// SPDX-License-Identifier: MIT
/// Part of numerics, (c) 2026 Aditya Dendukuri.
/// https://github.com/AdityaDendukuri/numerics
///
/// Separate so the compute headers never pull `<ostream>`. The operator is in `num::kernel`, so
/// ADL finds it.
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
