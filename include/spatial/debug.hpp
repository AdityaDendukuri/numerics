/// @file spatial/debug.hpp
/// @brief Runtime checks of spatial structures and kernels.
#pragma once

#include "core/debug.hpp"
#include "core/types.hpp"
#include "spatial/concepts.hpp"
#include <cmath>
#include <source_location>
#include <string>

namespace num::spatial::debug {

using num::debug::diagnostic_level;
using num::debug::get_level;
using num::debug::panic;

/// @brief Check that a kernel integrates to one: \f$\int_0^{2h} W(r, h)\, 2\pi r\, dr = 1\f$.
template <class K, class Scalar = float>
requires smoothing_kernel<K, Scalar> inline void
verify_kernel_normalization(Scalar h, Scalar tol = Scalar(2e-2),
                            std::source_location loc = std::source_location::current()) {
    if constexpr (!num::debug::sampling_compiled_in) {
        return;
    }
    if (num::debug::get_level() != num::debug::diagnostic_level::full) {
        return;
    }
    const int panels = 4000;
    const Scalar upper = Scalar(2) * h;
    const Scalar step = upper / static_cast<Scalar>(panels);
    Scalar total = Scalar(0);
    for (int i = 0; i < panels; ++i) {
        const Scalar r = (static_cast<Scalar>(i) + Scalar(0.5)) * step;
        total += K::W(r, h) * Scalar(2) * Scalar(3.14159265358979) * r * step;
    }
    if (std::abs(total - Scalar(1)) > tol) {
        panic("KernelError",
              "smoothing kernel is not normalized: integral over the support is " +
                  std::to_string(static_cast<double>(total)) +
                  " rather than 1, so SPH sums will not reproduce a constant field.",
              loc);
    }
}

/// @brief Check that a kernel vanishes outside its support: \f$W(r, h) = 0\f$ for \f$r > 2h\f$.
template <class K, class Scalar = float>
requires smoothing_kernel<K, Scalar> inline void
verify_kernel_support(Scalar h, std::source_location loc = std::source_location::current()) {
    if (get_level() == diagnostic_level::off) {
        return;
    }
    const Scalar outside = Scalar(2.001) * h;
    if (K::W(outside, h) != Scalar(0)) {
        panic("KernelError",
              "smoothing kernel is nonzero beyond r = 2h, so a neighbour search "
              "cut at that radius will miss contributions.",
              loc);
    }
}

/// @brief Check lattice neighbours: up then down, and left then right, return to the start.
template <class P>
requires periodic_neighbourhood_2d<P> inline void
verify_lattice_symmetry(const P &lattice,
                        std::source_location loc = std::source_location::current()) {
    if (get_level() == diagnostic_level::off) {
        return;
    }
    const int sites = lattice.N * lattice.N;
    for (int i = 0; i < sites; ++i) {
        if (lattice.dn[lattice.up[i]] != i || lattice.up[lattice.dn[i]] != i ||
            lattice.rt[lattice.lt[i]] != i || lattice.lt[lattice.rt[i]] != i) {
            panic("LatticeError",
                  "periodic neighbour tables are not mutually inverse at site " + std::to_string(i),
                  loc);
        }
    }
}

} // namespace num::spatial::debug
