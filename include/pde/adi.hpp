/// @file pde/adi.hpp
/// @brief Crank-Nicolson ADI sweeps for 2D parabolic systems.
/// @todo Add real-valued diffusion ADI variants, variable coefficients, and
/// boundary-condition parameterization.
#pragma once

#include "container/vector.hpp"
#include "linear/factorization/tridiag_complex.hpp"
#include "pde/stencil.hpp"
#include <complex>
#include <vector>

namespace num {

struct crank_nicolson_adi {
    int N = 0;
    double dt = 0.0;
    double h = 0.0;

    crank_nicolson_adi() = default;

    crank_nicolson_adi(int N_, double dt_, double h_) : N(N_), dt(dt_), h(h_) {
        using cplx = std::complex<double>;
        auto factor = [&](double tau) {
            double alpha = tau / (4.0 * h * h);
            cplx a(0.0, -alpha);
            cplx b(1.0, 2.0 * alpha);
            complex_tri_diag td;
            td.factor(N, a, b, a);
            return td;
        };
        td_half_ = factor(dt * 0.5);
        td_full_ = factor(dt);
    }

    void sweep(cvec &psi, bool x_axis, double tau) const {
        using cplx = std::complex<double>;
        const complex_tri_diag &td = (tau < dt * 0.75) ? td_half_ : td_full_;
        const cplx ia(0.0, tau / (4.0 * h * h));
        const cplx diag(1.0, -2.0 * tau / (4.0 * h * h));

        auto apply = [&](std::vector<cplx> &fiber) {
            std::vector<cplx> rhs(N);
            for (int i = 0; i < N; ++i) {
                cplx prev = (i > 0) ? fiber[i - 1] : cplx{};
                cplx next = (i < N - 1) ? fiber[i + 1] : cplx{};
                rhs[i] = ia * prev + diag * fiber[i] + ia * next;
            }
            td.solve(rhs);
            fiber = std::move(rhs);
        };

        if (x_axis) {
            col_fiber_sweep(psi, N, apply);
        } else {
            row_fiber_sweep(psi, N, apply);
        }
    }

  private:
    complex_tri_diag td_half_;
    complex_tri_diag td_full_;
};

} // namespace num
