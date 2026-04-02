/// @file examples/heat_demo.cpp
/// @brief 2D heat equation: Gaussian blob spreading on [0,1]^2.
///
/// PDE:  u_t = kappa * nabla^2 u,  Dirichlet BCs (u = 0 on boundary).
///
/// Spatial: 2nd-order 5-point Laplacian (diffusion_step_2d_dirichlet).
/// Time:    explicit Euler, dt = 0.20 h^2  (stability limit: 0.25 h^2).
///
/// Output: three-panel heatmap saved to heat_demo.png
///         left: t = 0  |  centre: t = T/2  |  right: t = T

#include "numerics.hpp"
#include <cstdio>
#include <string>

int main() {
    const int    N     = 64;
    const double kappa = 1.0;
    const double sigma = 0.06;
    const double T_end = 0.012;
    const double h     = 1.0 / (N + 1);
    const double dt    = 0.20 * h * h;
    const int    nstep = static_cast<int>(T_end / dt) + 1;
    const double coeff = kappa * dt / (h * h);

    // Initial condition: Gaussian blob centred at (0.5, 0.5)
    num::Vector u(static_cast<std::size_t>(N) * N);
    num::fill_grid(u, N, h, [=](double x, double y) {
        return num::gaussian2d(x, y, 0.5, 0.5, sigma);
    });

    num::Vector u0   = u;
    num::Vector u_mid;

    for (int s = 0; s < nstep; ++s) {
        if (s == nstep / 4) { u_mid = u; }
        num::pde::diffusion_step_2d_dirichlet(u, N, coeff);
    }

    const double t_mid = (nstep / 4) * dt;

    num::plt::subplot(3, 1);

    num::plt::heatmap(u0, N, h);
    num::plt::title("t = 0");
    num::plt::xlabel("x");
    num::plt::ylabel("y");
    num::plt::next();

    num::plt::heatmap(u_mid, N, h);
    num::plt::title("t = " + std::to_string(t_mid).substr(0, 6));
    num::plt::xlabel("x");
    num::plt::next();

    num::plt::heatmap(u, N, h);
    num::plt::title("t = " + std::to_string(T_end).substr(0, 6));
    num::plt::xlabel("x");

    num::plt::savefig("heat_demo.png");
}
