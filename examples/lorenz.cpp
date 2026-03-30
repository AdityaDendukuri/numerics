/// @file examples/lorenz.cpp
/// @brief Lorenz attractor integrated with num::ode_rk45, plotted with num::plt.
///
/// The Lorenz system (1963):
///   dx/dt = sigma*(y - x)
///   dy/dt = x*(rho - z) - y
///   dz/dt = x*y - beta*z
///
/// With classical parameters (sigma=10, rho=28, beta=8/3) the system is chaotic.
/// num::ode_rk45 adapts the step size automatically throughout the integration.

#include "numerics.hpp"
#include <cstdio>

int main() {
    const double sigma = 10.0, rho = 28.0, beta = 8.0 / 3.0;

    auto lorenz = [&](double, const num::Vector& s, num::Vector& ds) {
        ds[0] = sigma * (s[1] - s[0]);
        ds[1] = s[0] * (rho - s[2]) - s[1];
        ds[2] = s[0] * s[1] - beta * s[2];
    };

    num::Series xz;
    xz.reserve(200000);

    num::Vector y0 = {1.0, 0.0, 0.0};
    auto result = num::ode_rk45(lorenz, y0, 0.0, 50.0,
                                 1e-8, 1e-10, 1e-3, 2000000,
                                 [&](double, const num::Vector& s) {
                                     xz.emplace_back(s[0], s[2]);
                                 });

    printf("%zu steps,  final t = %.4f\n", (size_t)result.steps, result.t);

    num::plt::plot(xz);
    num::plt::title("Lorenz attractor  (sigma=10, rho=28, beta=8/3)");
    num::plt::xlabel("x");
    num::plt::ylabel("z");
    num::plt::show();
}
