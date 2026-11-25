/// @file spatial/sph_kernel.hpp
/// @brief SPH smoothing kernels in 2D and 3D.
#pragma once

#include "core/types.hpp"
#include "spatial/concepts.hpp"
#include <array>
#include <cmath>

namespace num {

namespace detail {

template <int Dim>
struct CubicSigma;
template <>
struct CubicSigma<2> {
    static float compute(float h) { return 10.0f / (7.0f * 3.14159265f * h * h); }
};
template <>
struct CubicSigma<3> {
    static float compute(float h) { return 1.0f / (3.14159265f * h * h * h); }
};

template <int Dim>
struct SpikyDW;
template <>
struct SpikyDW<2> {
    static float compute(float r, float h) {
        const float H = 2.0f * h;
        if (r >= H) {
            return 0.0f;
        }
        const float h5 = h * h * h * h * h;
        const float d = H - r;
        return (-15.0f / (16.0f * 3.14159265f * h5)) * d * d;
    }
};
template <>
struct SpikyDW<3> {
    static float compute(float r, float h) {
        const float H = 2.0f * h;
        if (r >= H || r < 1e-10f) {
            return 0.0f;
        }
        const float H6 = H * H * H * H * H * H;
        const float d = H - r;
        return -45.0f / (3.14159265f * H6) * d * d;
    }
};

} // namespace detail

/// @brief SPH kernels: Monaghan cubic spline \f$W(r, h)\f$ and Desbrun spiky gradient.
/// Support is \f$r \le 2h\f$.
template <int Dim>
struct sph_kernel {
    static_assert(Dim == 2 || Dim == 3, "sph_kernel: Dim must be 2 or 3");

    /// @brief Cubic spline kernel: \f$W(r, h)\f$, \f$\int W\, dV = 1\f$.
    static float W(float r, float h) {
        const float sigma = detail::CubicSigma<Dim>::compute(h);
        const float q = r / h;
        if (q <= 1.0f) {
            return sigma * (1.0f - (1.5f * q * q) + (0.75f * q * q * q));
        }
        if (q <= 2.0f) {
            const float t = 2.0f - q;
            return sigma * 0.25f * t * t * t;
        }
        return 0.0f;
    }

    /// @brief Cubic spline radial derivative: \f$\partial W / \partial r\f$.
    static float dW_dr(float r, float h) {
        const float sigma = detail::CubicSigma<Dim>::compute(h);
        const float q = r / h;
        if (q <= 1.0f) {
            return (sigma / h) * ((-3.0f * q) + (2.25f * q * q));
        }
        if (q <= 2.0f) {
            const float t = 2.0f - q;
            return (sigma / h) * (-0.75f * t * t);
        }
        return 0.0f;
    }

    /// @brief Spiky kernel radial derivative: \f$\partial W_{\mathrm{spiky}} / \partial r\f$.
    static float Spiky_dW_dr(float r, float h) { return detail::SpikyDW<Dim>::compute(r, h); }

    /// @brief Spiky kernel gradient: \f$\nabla W_{\mathrm{spiky}}(r_{ij}, h)\f$.
    static std::array<float, Dim> Spiky_gradW(std::array<float, Dim> r_vec, float r, float h) {
        std::array<float, Dim> g{};
        if (r < 1e-10f || r >= 2.0f * h) {
            return g;
        }
        const float c = detail::SpikyDW<Dim>::compute(r, h) / r;
        for (int d = 0; d < Dim; ++d) {
            g[d] = c * r_vec[d];
        }
        return g;
    }
};

static_assert(smoothing_kernel<sph_kernel<2>, float>);
static_assert(smoothing_kernel<sph_kernel<3>, float>);
} // namespace num
