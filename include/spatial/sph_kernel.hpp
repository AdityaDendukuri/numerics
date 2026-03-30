/// @file spatial/sph_kernel.hpp
/// @brief Dimension-generic SPH smoothing kernels.
///
/// SPHKernel<2>  --  2D kernels  (cubic sigma = 10/(7*pi*h^2), spiky = -15/(16*pi*h^5) (2h-r)^2)
/// SPHKernel<3>  --  3D kernels  (cubic sigma = 1/(pi*h^3),   spiky = -45/(pi*(2h)^6) (2h-r)^2)
///
/// Support radius 2h throughout; q = r/h.
///
/// Cubic spline (density + Morris viscosity Laplacian):
///   W = sigma * { 1 - 1.5q^2 + 0.75q^3   q <= 1
///              { 0.25(2-q)^3            1 < q <= 2
///   dW/dr = sigma/h * { -3q + 2.25q^2    q <= 1
///                     { -0.75(2-q)^2    1 < q <= 2
///
/// Spiky kernel (pressure gradient  -- dW/dr != 0 as r->0 prevents clustering):
///   dW/dr < 0,  Spiky_gradW returns (dW/dr / r) * r_vec
///
/// Spiky_gradW takes std::array<float, Dim> and returns std::array<float, Dim>.
#pragma once

#include <cmath>
#include <array>

namespace num {

namespace detail {

template<int Dim> struct CubicSigma;
template<> struct CubicSigma<2> {
    static float compute(float h) {
        return 10.0f / (7.0f * 3.14159265f * h * h);
    }
};
template<> struct CubicSigma<3> {
    static float compute(float h) {
        return 1.0f / (3.14159265f * h * h * h);
    }
};

template<int Dim> struct SpikyDW;
template<> struct SpikyDW<2> {
    // dW/dr = -15/(16*pi*h^5) * (2h-r)^2
    static float compute(float r, float h) {
        const float H = 2.0f * h;
        if (r >= H) return 0.0f;
        const float h5 = h * h * h * h * h;
        const float d  = H - r;
        return (-15.0f / (16.0f * 3.14159265f * h5)) * d * d;
    }
};
template<> struct SpikyDW<3> {
    // dW/dr = -45/(pi*H^6) * (H-r)^2,  H = 2h
    static float compute(float r, float h) {
        const float H = 2.0f * h;
        if (r >= H || r < 1e-10f) return 0.0f;
        const float H6 = H * H * H * H * H * H;
        const float d  = H - r;
        return -45.0f / (3.14159265f * H6) * d * d;
    }
};

} // namespace detail

/// Dimension-generic SPH smoothing kernels. Dim = 2 or 3.
template<int Dim>
struct SPHKernel {
    static_assert(Dim == 2 || Dim == 3, "SPHKernel: Dim must be 2 or 3");

    /// 2D/3D cubic spline density kernel.  Support = 2h.
    static float W(float r, float h) {
        const float sigma = detail::CubicSigma<Dim>::compute(h);
        const float q = r / h;
        if (q <= 1.0f)
            return sigma * (1.0f - 1.5f * q * q + 0.75f * q * q * q);
        if (q <= 2.0f) {
            const float t = 2.0f - q;
            return sigma * 0.25f * t * t * t;
        }
        return 0.0f;
    }

    /// Radial derivative dW/dr of cubic spline (<= 0 for r > 0).
    /// Used for the Morris SPH Laplacian in viscosity and heat.
    static float dW_dr(float r, float h) {
        const float sigma = detail::CubicSigma<Dim>::compute(h);
        const float q = r / h;
        if (q <= 1.0f) return (sigma / h) * (-3.0f * q + 2.25f * q * q);
        if (q <= 2.0f) {
            const float t = 2.0f - q;
            return (sigma / h) * (-0.75f * t * t);
        }
        return 0.0f;
    }

    /// Radial derivative dW/dr of spiky kernel (<= 0, non-zero at r=0).
    static float Spiky_dW_dr(float r, float h) {
        return detail::SpikyDW<Dim>::compute(r, h);
    }

    /// Gradient of spiky kernel: g = (dW/dr / r) * r_vec.
    /// Returns zero array if r < eps or r >= 2h.
    static std::array<float, Dim> Spiky_gradW(std::array<float, Dim> r_vec,
                                               float r, float h) {
        std::array<float, Dim> g{};
        if (r < 1e-10f || r >= 2.0f * h) return g;
        const float c = detail::SpikyDW<Dim>::compute(r, h) / r;
        for (int d = 0; d < Dim; ++d) g[d] = c * r_vec[d];
        return g;
    }
};

} // namespace num
