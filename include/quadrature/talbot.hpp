/// @file quadrature/talbot.hpp
/// @brief Weideman–Talbot contours for numerical inverse Laplace transforms.
#pragma once

#include "core/types.hpp"
#include "quadrature/concepts.hpp"
#include <cmath>
#include <complex>
#include <stdexcept>
#include <vector>

namespace num {

/// @brief Contour node: \f$f(t) \approx \sum_k w_k F(s_k)\f$, weights including \f$1 / 2\pi i\f$.
struct contour_node {
    cplx shift;
    cplx weight;
};

/// @brief Weideman–Talbot quadrature: \f$f(t) = \mathcal{L}^{-1}[F](t)\f$.
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

/// @brief Weideman–Talbot contour for \f$f(t) = \mathcal{L}^{-1}[F](t)\f$, \f$t > 0\f$.
inline array<contour_node> talbot_contour(real t, idx modes = 16) {
    return talbot_quadrature{modes}.nodes(t);
}

/// @brief Half Weideman–Talbot contour for real \f$f\f$: \f$F(\bar{s}) = \overline{F(s)}\f$.
/// Each weight is doubled.
inline array<contour_node> real_talbot_contour(real t, idx modes = 16) {
    if (modes % 2 != 0) {
        throw std::invalid_argument("real_talbot_contour requires an even mode count");
    }
    array<contour_node> nodes = talbot_contour(t, modes);
    array<contour_node> half;
    half.reserve(modes / 2);
    for (idx k = modes / 2; k < modes; ++k) {
        nodes[k].weight *= 2.0;
        half.push_back(nodes[k]);
    }
    return half;
}

static_assert(contour_rule<talbot_quadrature>);
} // namespace num
