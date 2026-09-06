/// @file spectral/debug.hpp
/// @brief Runtime verification of transform properties.
#pragma once

#include "container/vector.hpp"
#include "core/debug.hpp"
#include "core/types.hpp"
#include "spectral/concepts.hpp"
#include <cmath>
#include <source_location>
#include <string>

namespace num::spectral::debug {

using num::debug::diagnostic_level;
using num::debug::get_level;
using num::debug::panic;

/// @brief Verify Parseval's identity for a forward transform.
///
/// \f[ \sum_n |x_n|^2 = \frac{1}{N} \sum_k |X_k|^2 \f]
///
/// A transform that violates it is not unitary after normalization, and a
/// spectral method built on it will gain or lose energy every step.
template <class Plan>
requires transform_plan<Plan> inline void
verify_parseval(const Plan &plan, real tol = 1e-10,
                std::source_location loc = std::source_location::current()) {
    if constexpr (!num::debug::sampling_compiled_in) {
        return;
    }
    if (num::debug::get_level() != num::debug::diagnostic_level::full) {
        return;
    }
    const int n = plan.size();
    if (n <= 0) {
        return;
    }

    cvec in(static_cast<idx>(n));
    cvec out(static_cast<idx>(n));
    std::uint64_t state = 0x9E3779B97F4A7C15ULL;
    const auto next = [&state]() {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        return ((static_cast<double>(state >> 11) / 9007199254740992.0) * 2.0) - 1.0;
    };
    for (idx i = 0; i < in.size(); ++i) {
        in[i] = cplx(next(), next());
    }

    plan.execute(in, out);

    real energy_time = 0.0;
    real energy_freq = 0.0;
    for (idx i = 0; i < in.size(); ++i) {
        energy_time += std::norm(in[i]);
        energy_freq += std::norm(out[i]);
    }
    energy_freq /= static_cast<real>(n);

    const real scale = std::max(energy_time, energy_freq) + std::numeric_limits<real>::min();
    if (std::abs(energy_time - energy_freq) / scale > tol) {
        panic("TransformError",
              "Parseval's identity does not hold: time-domain energy " +
                  std::to_string(energy_time) + " against frequency-domain energy " +
                  std::to_string(energy_freq) +
                  ". The transform is not unitary after normalization.",
              loc);
    }
}

} // namespace num::spectral::debug
