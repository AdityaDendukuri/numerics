/// @file 08_root_finding_and_quadrature.cpp
/// @brief Newton-Raphson, Bisection, Gauss-Legendre & Adaptive Simpson Integration.
#include <cmath>
#include <iostream>
#include <numerics.hpp>

int main() {
    using namespace num;

    // 1. Root Finding: f(x) = x^2 - 2.0 = 0 -> sqrt(2)
    auto f = [](real x) { return (x * x) - 2.0; };
    auto df = [](real x) { return 2.0 * x; };

    auto root_newton = newton(f, df, 1.0, 1e-10, 100);
    auto root_bisect = bisection(f, 1.0, 2.0, 1e-10, 100);
    std::cout << "Newton Root: " << root_newton.root << " | Bisection Root: " << root_bisect.root
              << "\n";

    // Plot target function f(x) over [0, 2]
    std::vector<double> x_val, f_val;
    for (int i = 0; i <= 40; ++i) {
        double x = i * 0.05;
        x_val.push_back(x);
        f_val.push_back(f(x));
    }

    plt::plot(x_val, f_val, "f(x) = x^2 - 2", "lines");
    plt::title("08 Root Finding: Function Curve f(x) with Root at sqrt(2)");
    plt::xlabel("x");
    plt::ylabel("f(x)");
    plt::show_dumb(140, 35);

    return 0;
}
