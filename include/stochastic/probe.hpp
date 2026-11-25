/// @file stochastic/probe.hpp
/// @brief Random probes for trace and diagonal estimation.
#pragma once

#include "container/matrix.hpp"
#include "container/vector.hpp"
#include "core/types.hpp"
#include "stochastic/rng.hpp"
#include <random>
#include <stdexcept>

namespace num {

/// @brief Rademacher probes: \f$Z \in \{\pm 1\}^{n \times p}\f$, \f$\mathbb{E}[z_k z_k^T] = I\f$.
template <typename RNG = rng>
[[nodiscard]] mat<real> rademacher_probe(idx n, idx probes, RNG &generator) {
    if (probes == 0) {
        throw std::invalid_argument("rademacher_probe: at least one probe is required");
    }
    std::bernoulli_distribution sign(0.5);
    mat<real> probe(n, probes, 0.0);
    for (idx j = 0; j < n; ++j) {
        for (idx p = 0; p < probes; ++p) {
            probe(j, p) = sign(generator) ? 1.0 : -1.0;
        }
    }
    return probe;
}

/// @brief Rademacher probes from a fixed seed.
[[nodiscard]] inline mat<real> rademacher_probe(idx n, idx probes, unsigned seed) {
    rng generator(seed);
    return rademacher_probe(n, probes, generator);
}

/// @brief Gaussian probes: \f$Z \in \mathbb{R}^{n \times p}\f$, entries \f$\mathcal{N}(0, 1)\f$.
template <typename RNG = rng>
[[nodiscard]] mat<real> gaussian_probe(idx n, idx probes, RNG &generator) {
    if (probes == 0) {
        throw std::invalid_argument("gaussian_probe: at least one probe is required");
    }
    std::normal_distribution<real> normal(0.0, 1.0);
    mat<real> probe(n, probes, 0.0);
    for (idx j = 0; j < n; ++j) {
        for (idx p = 0; p < probes; ++p) {
            probe(j, p) = normal(generator);
        }
    }
    return probe;
}

/// @brief Gaussian probes from a fixed seed.
[[nodiscard]] inline mat<real> gaussian_probe(idx n, idx probes, unsigned seed) {
    rng generator(seed);
    return gaussian_probe(n, probes, generator);
}

/// @brief Hutchinson estimate: \f$\mathrm{diag}(B B^T) \approx \frac{1}{p} \sum_k (B z_k)^2\f$.
[[nodiscard]] inline vec<real> hutchinson_row_mean_square(const mat<real> &probed) {
    if (probed.cols() == 0) {
        throw std::invalid_argument("hutchinson_row_mean_square: no probed columns");
    }
    vec<real> estimate(probed.rows(), 0.0);
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
