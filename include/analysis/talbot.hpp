/// @file talbot.hpp
/// @brief Domain-independent midpoint Weideman--Talbot contour utilities.
#pragma once

#include "core/types.hpp"
#include <cmath>
#include <complex>
#include <stdexcept>
#include <vector>

namespace num {

struct TalbotNode {
    cplx shift;
    cplx weight;
};

/// Return quadrature nodes and weights for f(t)=L^{-1}[F](t), t>0.
/// The contour is scaled per requested time; weights include 1/(2 pi i).
inline std::vector<TalbotNode> talbot_nodes(real t, idx modes = 16) {
    if (!(t > 0.0) || modes < 2) {
        throw std::invalid_argument("talbot_nodes: invalid time or mode count");
    }
    constexpr real sigma = 0.6407;
    constexpr real mu = 0.5017;
    constexpr real nu = 0.6122;
    constexpr real eta = 0.2645;
    std::vector<TalbotNode> nodes;
    nodes.reserve(modes);
    const real pi = std::acos(-1.0);
    for (idx k = 0; k < modes; ++k) {
        const real theta = -pi + ((static_cast<real>(k) + 0.5) * (2.0 * pi / modes));
        const real a = sigma * theta;
        const real cot = std::cos(a) / std::sin(a);
        const real csc2 = 1.0 / (std::sin(a) * std::sin(a));
        const real re = (mu * theta * cot) - nu;
        const real dre = mu * (cot - (a * csc2));
        const real im = eta * theta;
        const real dim = eta;
        // The Weideman contour is scaled by the number of quadrature modes.
        // Keep this factor in both the node and its derivative; otherwise the
        // resolvent samples are at the wrong frequencies and inversion is biased.
        const real scale = static_cast<real>(modes);
        const cplx z(scale * re, scale * im);
        const cplx dz(scale * dre, scale * dim);
        nodes.push_back(
            {z / t, std::exp(z) * dz / (cplx(0.0, 1.0) * static_cast<real>(modes) * t)});
    }
    return nodes;
}

/// Drive inverse-Laplace accumulation without prescribing the transformed
/// value type. The callback receives each shift and fully scaled weight.
template <typename Accumulate>
void inverse_laplace_accumulate(real time, idx modes, Accumulate &&accumulate) {
    for (const TalbotNode &node : talbot_nodes(time, modes)) {
        accumulate(node.shift, node.weight);
    }
}

} // namespace num
