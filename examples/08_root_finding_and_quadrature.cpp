/// @file 08_root_finding_and_quadrature.cpp
/// @brief Root finding with bisection, Newton, secant and Brent, and five quadrature rules.
///
/// Each root finder returns a `root_result<T>` holding the root, the iterations taken and the final
/// residual. The quadrature rules integrate a callable over [a, b], and the table compares them on
/// \f$\int_0^\pi \sin x\,dx = 2\f$.
#include <cmath>
#include <cstdio>
#include <numbers>
#include <numerics.hpp>
#include <string_view>
#include <vector>

using namespace num;

int main(int argc, char **argv) {
    const bool plot = argc > 1 && std::string_view(argv[1]) == "--plot";

    // The root of x^2 - 2 on [1, 2] is sqrt(2).
    const auto f = [](real x) { return (x * x) - 2.0; };
    const auto df = [](real x) { return 2.0 * x; };
    const real root = std::sqrt(2.0);
    auto report = [&](const char *name, const root_result<real> &r) {
        std::printf("%-10s root %.12f  error %.1e  iterations %2zu\n", name, r.root,
                    std::abs(r.root - root), static_cast<std::size_t>(r.iterations));
    };
    report("bisection", bisection(f, 1.0, 2.0, 1e-12, 100));
    report("newton", newton(f, df, 1.0, 1e-12, 100));
    report("secant", secant(f, 1.0, 2.0, 1e-12, 100));
    report("brent", brent(f, 1.0, 2.0, 1e-12, 100));

    const auto g = [](real x) { return std::sin(x); };
    const real pi = std::numbers::pi;
    std::printf("\nintegral of sin on [0, pi] = 2\n");
    std::printf("trapz, 100 points        error %.1e\n", std::abs(trapz(g, 0.0, pi, 100) - 2.0));
    std::printf("simpson, 100 points      error %.1e\n", std::abs(simpson(g, 0.0, pi, 100) - 2.0));
    std::printf("gauss_legendre, 5 nodes  error %.1e\n", std::abs(gauss_legendre(g, 0.0, pi, 5) - 2.0));
    std::printf("adaptive_simpson 1e-10   error %.1e\n",
                std::abs(adaptive_simpson(g, 0.0, pi, 1e-10) - 2.0));
    std::printf("romberg 1e-10            error %.1e\n", std::abs(romberg(g, 0.0, pi, 1e-10) - 2.0));

    if (plot) {
        std::vector<double> x, y;
        for (int i = 0; i <= 40; ++i) {
            x.push_back(i * 0.05);
            y.push_back(f(i * 0.05));
        }
        plt::plot(x, y, "x^2 - 2", "lines");
        plt::title("08 Root at sqrt(2)");
        plt::show();
    }
}
