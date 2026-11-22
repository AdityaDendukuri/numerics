/// @file 08_root_finding_and_quadrature.cpp
/// @brief Root finding with bisection, Newton, secant and Brent, and five quadrature rules.
///
/// Each root finder returns a `root_result<T>` holding the root, the iterations taken and the final
/// residual. The quadrature rules integrate a callable over [a, b], and the saved result compares them on
/// \f$\int_0^\pi \sin x\,dx = 2\f$.
#include <cmath>
#include <io/json.hpp>
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
    io::json roots;
    auto record = [&](const char *name, const root_result<real> &r) {
        roots[name] = {{"root", r.root}, {"error", std::abs(r.root - root)},
                       {"iterations", r.iterations}};
    };
    record("bisection", bisection(f, 1.0, 2.0, 1e-12, 100));
    record("newton", newton(f, df, 1.0, 1e-12, 100));
    record("secant", secant(f, 1.0, 2.0, 1e-12, 100));
    record("brent", brent(f, 1.0, 2.0, 1e-12, 100));

    const auto g = [](real x) { return std::sin(x); };
    const real pi = std::numbers::pi;
    io::write_json({
        {"roots", roots},
        {"quadrature_error", {
            {"trapz", std::abs(trapz(g, 0.0, pi, 100) - 2.0)},
            {"simpson", std::abs(simpson(g, 0.0, pi, 100) - 2.0)},
            {"gauss_legendre", std::abs(gauss_legendre(g, 0.0, pi, 5) - 2.0)},
            {"adaptive_simpson", std::abs(adaptive_simpson(g, 0.0, pi, 1e-10) - 2.0)},
            {"romberg", std::abs(romberg(g, 0.0, pi, 1e-10) - 2.0)}
        }}
    }, "result.json");

    if (plot) {
        std::vector<double> x, y;
        for (int i = 0; i <= 40; ++i) {
            x.push_back(i * 0.05);
            y.push_back(f(i * 0.05));
        }
        plt::plot(x, y, "x^2 - 2", "lines");
        plt::title("08 Root at sqrt(2)");
        plt::savefig("plot.png");
    }
}
