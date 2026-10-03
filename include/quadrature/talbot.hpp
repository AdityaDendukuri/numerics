/// @file quadrature/talbot.hpp
/// @brief The midpoint Weideman--Talbot contour for numerical inverse Laplace transforms.
#pragma once

#include "core/types.hpp"
#include "quadrature/concepts.hpp"
#include <cmath>
#include <complex>
#include <stdexcept>
#include <vector>

namespace num {

/// One node of an inversion contour: f(t) is approximately the sum of
/// `weight * F(shift)` over the nodes, with the weights including 1/(2 pi i).
struct contour_node {
    cplx shift;
    cplx weight;
};

/// @brief Weideman--Talbot hyperbolic contour quadrature for Numerical Inverse Laplace Transformation.
struct talbot_quadrature {
    idx modes = 16;
    real sigma = 0.6407;
    real mu = 0.5017;
    real nu = 0.6122;
    real eta = 0.2645;

    talbot_quadrature() = default;
    /* implicit */ talbot_quadrature(idx n_modes) : modes(n_modes) {}

    [[nodiscard]] array<contour_node> nodes(real t) const {
        if (!(t > 0.0) || modes < 2) {
            throw std::invalid_argument("talbot_quadrature: invalid time or mode count");
        }
        array<contour_node> result;
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
};

/// Return quadrature nodes and weights on the Weideman--Talbot inversion contour for \f$f(t) = \mathcal{L}^{-1}[F](t), \; t > 0\f$.
/// The contour is scaled per requested time; weights include \f$1/(2\pi i)\f$.
inline array<contour_node> talbot_contour(real t, idx modes = 16) {
    return talbot_quadrature{modes}.nodes(t);
}

static_assert(contour_rule<talbot_quadrature>);
} // namespace num
