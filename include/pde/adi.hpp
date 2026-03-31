/// @file pde/adi.hpp
/// @brief Crank-Nicolson ADI solver for 2D parabolic equations via fiber
/// sweeps.
///
/// CrankNicolsonADI encapsulates the prefactored complex tridiagonals for
/// Strang-split time stepping of equations of the form:
///
///   i*d(psi)/dt = -(1/2)*Lap(psi) + V(x,y)*psi   (Schrodinger)
///   d(u)/dt     = kappa*Lap(u)                     (complex-coefficient
///   diffusion)
///
/// Each CN sub-step tau along one axis solves a 1D system per fiber:
///
///   (I - ia*Lap_1D)*psi^{n+1} = (I + ia*Lap_1D)*psi^n,   ia = i*tau/(4*h^2)
///
/// Two tridiagonals are prefactored once -- for tau = dt/2 and tau = dt --
/// covering the full Strang splitting: sweep_x(dt/2) -> sweep_y(dt) ->
/// sweep_x(dt/2).
///
/// Typical usage (TDSE Strang splitting):
/// @code
///   num::CrankNicolsonADI adi(N, dt, h);
///
///   // one full time step:
///   adi.sweep(psi, true,  dt * 0.5);  // x, half-step
///   adi.sweep(psi, false, dt);         // y, full-step
///   adi.sweep(psi, true,  dt * 0.5);  // x, half-step
/// @endcode
#pragma once

#include "pde/stencil.hpp"
#include "linalg/factorization/tridiag_complex.hpp"
#include "core/vector.hpp"
#include <vector>
#include <complex>

namespace num {

struct CrankNicolsonADI {
    int    N  = 0;
    double dt = 0.0;
    double h  = 0.0;

    CrankNicolsonADI() = default;

    /// Pre-factor both tridiagonals.
    /// @param N_   Interior grid points per axis
    /// @param dt_  Full timestep
    /// @param h_   Grid spacing
    CrankNicolsonADI(int N_, double dt_, double h_)
        : N(N_)
        , dt(dt_)
        , h(h_) {
        using cplx = std::complex<double>;
        // alpha = tau/(4*h^2).  LHS tridiagonal: a=c=-i*alpha, b=1+2i*alpha.
        auto factor = [&](double tau) {
            double         alpha = tau / (4.0 * h * h);
            cplx           a(0.0, -alpha);
            cplx           b(1.0, 2.0 * alpha);
            ComplexTriDiag td;
            td.factor(N, a, b, a);
            return td;
        };
        td_half_ = factor(dt * 0.5);
        td_full_ = factor(dt);
    }

    /// Apply one CN sweep along x (col fibers, x_axis=true) or y (row fibers).
    /// Selects the half-step tridiagonal for tau < 0.75*dt, full-step
    /// otherwise.
    void sweep(CVector& psi, bool x_axis, double tau) const {
        using cplx               = std::complex<double>;
        const ComplexTriDiag& td = (tau < dt * 0.75) ? td_half_ : td_full_;
        const cplx            ia(0.0, tau / (4.0 * h * h));
        const cplx            diag(1.0, -2.0 * tau / (4.0 * h * h));

        auto apply = [&](std::vector<cplx>& fiber) {
            std::vector<cplx> rhs(N);
            for (int i = 0; i < N; ++i) {
                cplx prev = (i > 0) ? fiber[i - 1] : cplx{};
                cplx next = (i < N - 1) ? fiber[i + 1] : cplx{};
                rhs[i]    = ia * prev + diag * fiber[i] + ia * next;
            }
            td.solve(rhs);
            fiber = std::move(rhs);
        };

        if (x_axis)
            col_fiber_sweep(psi, N, apply);
        else
            row_fiber_sweep(psi, N, apply);
    }

  private:
    ComplexTriDiag td_half_;
    ComplexTriDiag td_full_;
};

} // namespace num
