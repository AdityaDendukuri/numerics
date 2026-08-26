/// @file quadrature/talbot.hpp
/// @brief Domain-independent midpoint Weideman--Talbot contour utilities.
#pragma once

#include "core/types.hpp"
#include <cmath>
#include <complex>
#include <stdexcept>
#include <vector>

namespace num {

struct ContourNode {
    cplx shift;
    cplx weight;
};

using TalbotNode = ContourNode;

/// @brief Weideman--Talbot hyperbolic contour quadrature for Numerical Inverse Laplace Transformation.
struct TalbotQuadrature {
    idx modes = 16;
    real sigma = 0.6407;
    real mu = 0.5017;
    real nu = 0.6122;
    real eta = 0.2645;

    TalbotQuadrature() = default;
    /* implicit */ TalbotQuadrature(idx n_modes) : modes(n_modes) {}

    [[nodiscard]] std::vector<ContourNode> nodes(real t) const {
        if (!(t > 0.0) || modes < 2) {
            throw std::invalid_argument("TalbotQuadrature: invalid time or mode count");
        }
        std::vector<ContourNode> result;
        result.reserve(modes);
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
            const real scale = static_cast<real>(modes);
            const cplx z(scale * re, scale * im);
            const cplx dz(scale * dre, scale * dim);
            result.push_back(
                {z / t, std::exp(z) * dz / (cplx(0.0, 1.0) * static_cast<real>(modes) * t)});
        }
        return result;
    }

    template <typename Accumulate>
    void accumulate(real t, Accumulate &&accumulate_fn) const {
        for (const ContourNode &node : nodes(t)) {
            accumulate_fn(node.shift, node.weight);
        }
    }
};

/// Return quadrature nodes and weights on the Weideman--Talbot inversion contour for \f$f(t) = \mathcal{L}^{-1}[F](t), \; t > 0\f$.
/// The contour is scaled per requested time; weights include \f$1/(2\pi i)\f$.
inline std::vector<ContourNode> talbot_contour(real t, idx modes = 16) {
    return TalbotQuadrature{modes}.nodes(t);
}

inline std::vector<ContourNode> talbot_nodes(real t, idx modes = 16) {
    return talbot_contour(t, modes);
}

/// Drive inverse-Laplace accumulation without prescribing the transformed
/// value type. The callback receives each shift and fully scaled weight.
template <typename Accumulate>
void inverse_laplace_accumulate(real time, idx modes, Accumulate &&accumulate) {
    TalbotQuadrature{modes}.accumulate(time, std::forward<Accumulate>(accumulate));
}

} // namespace num
