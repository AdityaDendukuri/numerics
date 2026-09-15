/// @file stochastic/probe.hpp
/// @brief Random probe vectors for stochastic trace and diagonal estimation.
#pragma once

#include "container/matrix.hpp"
#include "container/vector.hpp"
#include "core/types.hpp"
#include "stochastic/rng.hpp"
#include <random>
#include <stdexcept>

namespace num {

/// @brief An \f$n \times p\f$ matrix of independent Rademacher (\f$\pm 1\f$) probes.
///
/// Column \f$k\f$ is a probe vector \f$z_k\f$ with \f$E[z_k z_k^T] = I\f$, the
/// input to Hutchinson-type estimators of \f$\operatorname{tr}(A)\f$ and
/// \f$\operatorname{diag}(A)\f$.
template <typename RNG = rng>
[[nodiscard]] mat rademacher_probe(idx n, idx probes, RNG &generator) {
    if (probes == 0) {
        throw std::invalid_argument("rademacher_probe: at least one probe is required");
    }
    std::bernoulli_distribution sign(0.5);
    mat probe(n, probes, 0.0);
    for (idx j = 0; j < n; ++j) {
        for (idx p = 0; p < probes; ++p) {
            probe(j, p) = sign(generator) ? 1.0 : -1.0;
        }
    }
    return probe;
}

/// @brief The same probe from a fixed seed.
[[nodiscard]] inline mat rademacher_probe(idx n, idx probes, unsigned seed) {
    rng generator(seed);
    return rademacher_probe(n, probes, generator);
}

/// @brief Row-wise mean square of probed columns: the Hutchinson estimate.
///
/// If column \f$k\f$ of `probed` is \f$B z_k\f$ for Rademacher probes
/// \f$z_k\f$, entry \f$j\f$ of the result estimates \f$(B B^T)_{jj}\f$.
[[nodiscard]] inline vec hutchinson_row_mean_square(const mat &probed) {
    if (probed.cols() == 0) {
        throw std::invalid_argument("hutchinson_row_mean_square: no probed columns");
    }
    vec estimate(probed.rows(), 0.0);
    for (idx j = 0; j < probed.rows(); ++j) {
        real total = 0.0;
        for (idx p = 0; p < probed.cols(); ++p) {
            total += probed(j, p) * probed(j, p);
        }
        estimate[j] = total / static_cast<real>(probed.cols());
    }
    return estimate;
}

} // namespace num
