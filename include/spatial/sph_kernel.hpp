/// @file spatial/sph_kernel.hpp
/// @brief Dimension-generic SPH smoothing kernels.
#pragma once

#include "core/types.hpp"
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

/// @brief Smoothed Particle Hydrodynamics (SPH) smoothing kernel and gradient evaluations in 2D / 3D.
///
/// Implements standard Monaghan cubic spline kernel \f$W(r, h)\f$ and Desbrun Spiky pressure kernel gradients.
/// Compact support is strictly within \f$r \in [0, 2h]\f$.
///
/// @tparam Dim Spatial dimension (2 or 3).
template <int Dim>
struct sph_kernel {
    static_assert(Dim == 2 || Dim == 3, "sph_kernel: Dim must be 2 or 3");

    /// @brief Evaluate Monaghan cubic spline kernel \f$W(r, h)\f$ at distance \f$r\f$ with smoothing length \f$h\f$.
    /// @param r Radial distance between particles \f$\|\mathbf{r}_i - \mathbf{r}_j\|\f$.
    /// @param h Smoothing radius.
    /// @return Kernel value \f$W(r, h)\f$ normalized so that \f$\int W(r, h)\,\mathrm{d}V = 1\f$.
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

    /// @brief Evaluate radial derivative of the cubic spline kernel \f$\frac{\partial W}{\partial r}\f$.
    /// @param r Radial distance.
    /// @param h Smoothing radius.
    /// @return scalar derivative \f$\frac{\partial W}{\partial r}\f$.
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

    /// @brief Evaluate radial derivative of the Desbrun Spiky pressure kernel.
    /// @param r Radial distance.
    /// @param h Smoothing radius.
    /// @return Spiky kernel derivative \f$\frac{\partial W_{\text{spiky}}}{\partial r}\f$.
    static float Spiky_dW_dr(float r, float h) { return detail::SpikyDW<Dim>::compute(r, h); }

    /// @brief Evaluate Spiky pressure kernel gradient vector \f$\nabla W_{\text{spiky}}(\mathbf{r}_{ij}, h)\f$.
    /// @param r_vec Relative position vector \f$\mathbf{r}_{ij} = \mathbf{r}_i - \mathbf{r}_j\f$.
    /// @param r Norm \f$\|\mathbf{r}_{ij}\|\f$.
    /// @param h Smoothing radius.
    /// @return Gradient vector in \f$\mathbb{R}^{\text{Dim}}\f$.
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

} // namespace num
