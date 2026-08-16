/// @file 08_root_finding_and_quadrature.cpp
/// @brief Newton-Raphson, Bisection, Gauss-Legendre & Adaptive Simpson Integration.
#include <numerics.hpp>
#include <iostream>
#include <cmath>

int main() {
    using namespace num;

    // 1. Root Finding: f(x) = x^2 - 2.0 = 0 -> sqrt(2)
    auto f = [](real x) { return x * x - 2.0; };
    auto df = [](real x) { return 2.0 * x; };

    auto root_newton = newton(f, df, 1.0, 1e-10, 100);
    auto root_bisect = bisection(f, 1.0, 2.0, 1e-10, 100);
    std::cout << "Newton Root: " << root_newton.root << " | Bisection Root: " << root_bisect.root << "\n";

    // 2. Numerical Quadrature: int_0^pi sin(x) dx = 2.0
    auto target_f = [](real x) { return std::sin(x); };
    real quad_simp = adaptive_simpson(target_f, 0.0, M_PI, 1e-8, 10);
    real quad_gauss = gauss_legendre(target_f, 0.0, M_PI, 5);

    std::cout << "Adaptive Simpson Int = " << quad_simp << " | Gauss-Legendre Int = " << quad_gauss << "\n";

    return 0;
}
