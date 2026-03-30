/// @file src/sim.cpp
/// @brief 2-D incompressible Navier-Stokes -- Chorin projection, periodic MAC grid

#include "ns_demo/sim.hpp"

#include <cmath>
#include <chrono>

using namespace num;

namespace ns {

//  Construction

NSSolver::NSSolver(idx N_, real dt_, real nu_)
    : N(N_), h(1.0 / N_), dt(dt_), nu(nu_),
      u(N_ * N_, 0.0), v(N_ * N_, 0.0), p(N_ * N_, 0.0),
      u_star(N_ * N_, 0.0), v_star(N_ * N_, 0.0), rhs(N_ * N_, 0.0)
{}

//  Initial condition  -- double shear layer
//
//  Two horizontal shear bands centred at y = 0.25 and y = 0.75.
//  A small vertical perturbation seeds Kelvin-Helmholtz roll-up.
//  The initial field is analytically divergence-free.

void NSSolver::init_shear_layer(real rho, real delta) {
    const real two_pi = 2.0 * M_PI;

    for (idx i = 0; i < N; ++i) {
        for (idx j = 0; j < N; ++j) {
            // u[i,j] lives at (i*h, (j+1/2)*h)
            real y = (j + 0.5) * h;
            u[at(i, j)] = (y <= 0.5)
                ? std::tanh((y - 0.25) / rho)
                : std::tanh((0.75 - y) / rho);

            // v[i,j] lives at ((i+1/2)*h, j*h)
            real x = (i + 0.5) * h;
            v[at(i, j)] = delta * std::sin(two_pi * x);
        }
    }

    // Zero pressure (correct for divergence-free initial data)
    for (idx k = 0; k < N * N; ++k) p[k] = 0.0;
}

//  Top-level step

void NSSolver::step() {
    auto t0 = std::chrono::steady_clock::now();

    advect();
    if (nu > 0.0) apply_diffusion();
    build_rhs();
    solve_pressure();
    project();

    auto t1 = std::chrono::steady_clock::now();
    stats.total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
}

//  Semi-Lagrangian advection
//
//  For each MAC face, trace a particle backwards in time and interpolate
//  the velocity from the current field.  Unconditionally stable for any dt.

void NSSolver::advect() {
    auto t0 = std::chrono::steady_clock::now();

    // u[i,j] at (i*h, (j+1/2)*h)
    // Surrounding v-faces for the y-velocity at this point:
    //   v[i-1,j], v[i,j], v[i-1,j+1], v[i,j+1]
#pragma omp parallel for schedule(static) collapse(2)
    for (idx i = 0; i < N; ++i) {
        for (idx j = 0; j < N; ++j) {
            real uu = u[at(i, j)];
            real vu = 0.25 * (v[at(wm1(i), j     )] + v[at(i, j     )] +
                              v[at(wm1(i), wp1(j))] + v[at(i, wp1(j))]);

            real xb = i * h - dt * uu;
            real yb = (j + 0.5) * h - dt * vu;
            u_star[at(i, j)] = interp_u(xb, yb);
        }
    }

    // v[i,j] at ((i+1/2)*h, j*h)
    // Surrounding u-faces for the x-velocity at this point:
    //   u[i,j-1], u[i+1,j-1], u[i,j], u[i+1,j]
#pragma omp parallel for schedule(static) collapse(2)
    for (idx i = 0; i < N; ++i) {
        for (idx j = 0; j < N; ++j) {
            real vv = v[at(i, j)];
            real uv = 0.25 * (u[at(i,     wm1(j))] + u[at(wp1(i), wm1(j))] +
                              u[at(i,     j      )] + u[at(wp1(i), j      )]);

            real xb = (i + 0.5) * h - dt * uv;
            real yb = j * h - dt * vv;
            v_star[at(i, j)] = interp_v(xb, yb);
        }
    }

    auto t1 = std::chrono::steady_clock::now();
    stats.advect_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
}

//  Explicit viscosity (forward Euler, stable only when dt <= h^2/(4nu))

void NSSolver::apply_diffusion() {
    const double c = dt * nu / (h * h);
    const int    n = static_cast<int>(N);
    pde::diffusion_step_2d(u_star, n, c);
    pde::diffusion_step_2d(v_star, n, c);
}

//  RHS: rhs = -div(u*)/dt
//
//  Discrete divergence on the MAC grid:
//    div(u*)[i,j] = (u*[i+1,j] - u*[i,j]) / h  +  (v*[i,j+1] - v*[i,j]) / h
//
//  We form the positive-definite system (-Delta)p = -div(u*)/dt so that CG works.

void NSSolver::build_rhs() {
    const real scale = -1.0 / (h * dt);   // -1/(h*dt) so rhs = -div/dt

#pragma omp parallel for schedule(static) collapse(2)
    for (idx i = 0; i < N; ++i) {
        for (idx j = 0; j < N; ++j) {
            real div_ij = u_star[at(wp1(i), j)] - u_star[at(i, j)]
                        + v_star[at(i, wp1(j))] - v_star[at(i, j)];
            rhs[at(i, j)] = scale * div_ij;
        }
    }

    // Remove mean (periodic Poisson is singular; mean(div) = 0 analytically,
    // but floating-point errors accumulate  -- subtracting keeps CG well-posed).
    real sum = 0.0;
    for (idx k = 0; k < N * N; ++k) sum += rhs[k];
    real mean = sum / static_cast<real>(N * N);
    for (idx k = 0; k < N * N; ++k) rhs[k] -= mean;
}

//  Pressure solve: (-Delta)p = rhs   via CG
//
//  (-Delta)p[i,j] = (4p[i,j] - p[i+/-1,j] - p[i,j+/-1]) / h^2
//
//  The operator is positive semi-definite (null space = constants).
//  We initialise p = previous p (warm start) and subtract the mean after.

void NSSolver::solve_pressure() {
    auto t0 = std::chrono::steady_clock::now();

    const real inv_h2 = 1.0 / (h * h);
    const int  n      = static_cast<int>(N);

    // Matrix-free negative Laplacian via laplacian_stencil_2d_periodic (boundary-peeled,
    // auto-vectorisable inner loop) then negate and scale.
    auto neg_lap = [&](const Vector& pin, Vector& out) {
        laplacian_stencil_2d_periodic(pin, out, n);
        scale(out, -inv_h2);
    };

    auto result = num::cg_matfree(neg_lap, rhs, p, /*tol=*/1e-3, /*max_iter=*/100);
    stats.cg_iters    = result.iterations;
    stats.cg_residual = result.residual;

    // Remove mean from pressure (physics is unchanged; improves stability)
    real sum = 0.0;
    for (idx k = 0; k < N * N; ++k) sum += p[k];
    real mean = sum / static_cast<real>(N * N);
    for (idx k = 0; k < N * N; ++k) p[k] -= mean;

    auto t1 = std::chrono::steady_clock::now();
    stats.pressure_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
}

//  Projection: u = u* - dt*gradp
//
//  Gradient at MAC face (i,j):
//    (d_p/d_x) at u-face = (p[i,j] - p[i-1,j]) / h
//    (d_p/d_y) at v-face = (p[i,j] - p[i,j-1]) / h

void NSSolver::project() {
    auto t0 = std::chrono::steady_clock::now();

    const real c = dt / h;

#pragma omp parallel for schedule(static) collapse(2)
    for (idx i = 0; i < N; ++i) {
        for (idx j = 0; j < N; ++j) {
            u[at(i, j)] = u_star[at(i, j)] - c * (p[at(i, j)] - p[at(wm1(i), j)]);
            v[at(i, j)] = v_star[at(i, j)] - c * (p[at(i, j)] - p[at(i, wm1(j))]);
        }
    }

    auto t1 = std::chrono::steady_clock::now();
    stats.project_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
}

//  Diagnostics

real NSSolver::vorticity(idx i, idx j) const {
    // omega = d_v/d_x - d_u/d_y  at corner (i*h, j*h)
    // v[i,j] at ((i+1/2)h, j*h),  v[i-1,j] at ((i-1/2)h, j*h)
    // u[i,j] at (i*h, (j+1/2)h),  u[i,j-1] at (i*h, (j-1/2)h)
    real dvdx = (v[at(i, j)] - v[at(wm1(i), j)]) / h;
    real dudy = (u[at(i, j)] - u[at(i, wm1(j))]) / h;
    return dvdx - dudy;
}

real NSSolver::speed(idx i, idx j) const {
    // Average faces to cell centre (i+1/2, j+1/2)*h
    real uc = 0.5 * (u[at(i, j)] + u[at(wp1(i), j)]);
    real vc = 0.5 * (v[at(i, j)] + v[at(i, wp1(j))]);
    return std::sqrt(uc * uc + vc * vc);
}

} // namespace ns
