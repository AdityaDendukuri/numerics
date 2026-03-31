/// @file examples/lorenz.cpp
/// @brief Lorenz attractor integrated with num::rk45, plotted with num::plt.
///
/// The Lorenz system (1963):
///   dx/dt = sigma*(y - x)
///   dy/dt = x*(rho - z) - y
///   dz/dt = x*y - beta*z
///
/// With classical parameters (sigma=10, rho=28, beta=8/3) the system is chaotic.
/// num::rk45 adapts the step size automatically throughout the integration.

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

    num::ODEParams params = {.tf = 50.0, .rtol = 1e-8, .atol = 1e-10, .max_steps = 2000000};

    // Runge-Kutta RK4(5): t=[0, 50], rtol=1e-8, atol=1e-10 -- each iteration is one accepted Step {t, y}.
    for (auto [t, y] : num::rk45(lorenz, y0, params))
        xz.emplace_back(y[0], y[2]);

    printf("%zu steps\n", xz.size());

    num::plt::plot(xz);
    num::plt::title("Lorenz attractor  (sigma=10, rho=28, beta=8/3)");
    num::plt::xlabel("x");
    num::plt::ylabel("z");
    num::plt::show();
}
