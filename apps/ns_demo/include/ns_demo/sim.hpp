/// @file include/ns_demo/sim.hpp
/// @brief 2-D incompressible Navier-Stokes, periodic MAC grid
///
/// Algorithm: Chorin projection method
///   1. Semi-Lagrangian advection  -> u*, v*
///   2. Build RHS: rhs = -div(u*)/dt
///   3. CG pressure solve: (-Lap)p = rhs   (positive-definite, Backend::omp
///   inner products)
///   4. Project: u = u* - dt*gradp
///
/// Grid (NxN cells, domain [0,1]^2):
///   u[i,j]   -- x-velocity at face  (i*h, (j+1/2)*h)    i,j in [0,N)
///   v[i,j]   -- y-velocity at face  ((i+1/2)*h, j*h)     i,j in [0,N)
///   p[i,j]   -- pressure at centre  ((i+1/2)*h, (j+1/2)*h)
///
/// Storage: row-major, index = i*N + j.
/// All boundaries are periodic.
#pragma once

#include "numerics.hpp"

#include <chrono>
#include <cmath>

namespace ns {

using num::idx;
using num::real;
using num::SolverResult;
using num::Vector;

struct Stats {
    idx    cg_iters    = 0;
    real   cg_residual = 0.0;
    double advect_ms   = 0.0;
    double pressure_ms = 0.0;
    double project_ms  = 0.0;
    double total_ms    = 0.0;
};

class NSSolver {
  public:
    /// @param N_   Grid resolution (NxN cells)
    /// @param dt_  Time step
    /// @param nu_  Kinematic viscosity (0 = inviscid Euler)
    NSSolver(idx N_, real dt_, real nu_ = 0.0);

    /// Double shear layer initial condition (Kelvin-Helmholtz instability).
    /// Two counter-flowing bands at y~=0.25 and y~=0.75 seed vortex roll-up.
    /// @param rho    Shear layer thickness (default 0.05)
    /// @param delta  Perturbation amplitude (default 0.05)
    void init_shear_layer(real rho = 0.05, real delta = 0.05);

    /// Advance one time step (advect -> pressure -> project).
    void step();

    /// Vorticity omega = d_v/d_x - d_u/d_y at grid corner (i*h, j*h).
    real vorticity(idx i, idx j) const;

    /// Velocity magnitude averaged to cell centre (i,j).
    real speed(idx i, idx j) const;

    /// Interpolate x-velocity at physical point (px, py).
    /// u[i,j] lives at (i*h, (j+1/2)*h).
    real interp_u(real px, real py) const {
        return num::sample_2d_periodic(u, N, h, px, py, 0.0, 0.5 * h);
    }
    /// Interpolate y-velocity at physical point (px, py).
    /// v[i,j] lives at ((i+1/2)*h, j*h).
    real interp_v(real px, real py) const {
        return num::sample_2d_periodic(v, N, h, px, py, 0.5 * h, 0.0);
    }

    // Grid constants
    const idx  N;
    const real h, dt, nu;

    // State
    Vector u, v, p; ///< velocity faces + cell-centre pressure, N*N each

    Stats stats;

  private:
    Vector u_star, v_star, rhs;

    // Index helpers
    inline idx at(idx i, idx j) const noexcept {
        return i * N + j;
    }
    inline idx wp1(idx i) const noexcept {
        return (i + 1) % N;
    }
    inline idx wm1(idx i) const noexcept {
        return (i + N - 1) % N;
    }

    // Physics sub-steps
    void advect();
    void apply_diffusion();
    void build_rhs();
    void solve_pressure();
    void project();
};

} // namespace ns
