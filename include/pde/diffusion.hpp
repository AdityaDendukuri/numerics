/// @file pde/diffusion.hpp
/// @brief Explicit Euler diffusion steps for 2D uniform grids.
///
/// diffusion_step_2d           -- u += coeff * Lap_periodic(u),  periodic BC
/// diffusion_step_2d_dirichlet -- u += coeff * Lap_dirichlet(u), Dirichlet BC
///
/// coeff = alpha*dt/h^2 where alpha is the diffusion coefficient.
///
/// Typical usage (viscosity in Navier-Stokes):
/// @code
///   const double coeff = dt * nu / (h * h);
///   num::pde::diffusion_step_2d(u_star, N, coeff, num::best_backend);
///   num::pde::diffusion_step_2d(v_star, N, coeff, num::best_backend);
/// @endcode
#pragma once

#include "pde/stencil.hpp"
#include "core/vector.hpp"
#include "core/policy.hpp"

namespace num::pde {

/// Explicit Euler diffusion step on a periodic 2D grid:
///   u += coeff * Lap_periodic(u)
///
/// @param u      NxN field vector (row-major)
/// @param N      Grid side length
/// @param coeff  Diffusion coefficient * dt / h^2 (e.g. nu*dt/h^2 for
/// viscosity)
/// @param b      Backend for the axpy accumulation
inline void diffusion_step_2d(Vector& u,
                              int     N,
                              double  coeff,
                              Backend b = best_backend) {
    Vector lap(u.size());
    laplacian_stencil_2d_periodic(u, lap, N);
    axpy(coeff, lap, u, b);
}

/// Explicit Euler diffusion step with Dirichlet (zero) BCs:
///   u += coeff * Lap_dirichlet(u)
inline void diffusion_step_2d_dirichlet(Vector& u,
                                        int     N,
                                        double  coeff,
                                        Backend b = best_backend) {
    Vector lap(u.size());
    laplacian_stencil_2d(u, lap, N);
    axpy(coeff, lap, u, b);
}

/// Explicit Euler diffusion step with Dirichlet BCs and 4th-order Laplacian:
///   u += coeff * Lap4_dirichlet(u)
///
/// coeff = alpha*dt/h^2. Stability requires coeff <= ~0.08 (vs 0.25 for 2nd order)
/// due to the larger spectral radius of the 13-point stencil.
inline void diffusion_step_2d_4th_dirichlet(Vector& u,
                                            int     N,
                                            double  coeff,
                                            Backend b = best_backend) {
    Vector lap(u.size());
    laplacian_stencil_2d_4th(u, lap, N);
    axpy(coeff, lap, u, b);
}

} // namespace num::pde
