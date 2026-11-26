/// @file quadrature/gauss_legendre_rule.hpp
/// @brief Gauss–Legendre nodes and weights of any order.
#pragma once

#include "linear/eigen/jacobi_eig.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

namespace num {

/// @brief Compute the Gauss–Legendre nodes \f$x_k\f$ and weights \f$w_k\f$ on \f$[a, b]\f$
/// (Golub–Welsch).
/// The weights sum to \f$b - a\f$.
inline std::pair<std::vector<real>, std::vector<real>> gauss_legendre_rule(idx order, real a = 0.0,
                                                                          real b = 1.0) {
    if (order == 0) {
        throw std::invalid_argument("gauss_legendre_rule: order must be positive");
    }
    mat<real> jacobi(order, order, 0.0);
    for (idx k = 1; k < order; ++k) {
        jacobi(k - 1, k) = jacobi(k, k - 1) =
            static_cast<real>(k) / std::sqrt(4.0 * static_cast<real>(k) * k - 1.0);
    }
    const auto eig = eig_sym(assume_symmetric(jacobi));
    const real midpoint = 0.5 * (a + b), halfwidth = 0.5 * (b - a);
    std::vector<real> nodes(order), weights(order);
    for (idx k = 0; k < order; ++k) {
        nodes[k] = midpoint + halfwidth * eig.values[k];
        weights[k] = (b - a) * eig.vectors(0, k) * eig.vectors(0, k);
    }
    return {nodes, weights};
}

} // namespace num
