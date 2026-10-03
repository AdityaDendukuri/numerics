/// @file 06_pde_poisson_solver.cpp
/// @brief A 3D Poisson problem on scalar fields, converging at second order.
///
/// `field_solver::solve_poisson(phi, s)` solves \f$\Delta\phi = s\f$ with zero Dirichlet
/// boundaries. It runs CG on the 7-point Laplacian and uses the field's own storage as the solution
/// vector. The manufactured solution \f$\phi = e^x\sin\pi x\sin\pi y\sin\pi z\f$ vanishes on the
/// boundary, and halving the grid spacing divides the error by about four.
#include <cmath>
#include <cstdio>
#include <numbers>
#include <numerics.hpp>
#include <string_view>
#include <vector>

using namespace num;

int main(int argc, char **argv) {
    const bool plot = argc > 1 && std::string_view(argv[1]) == "--plot";
    constexpr real pi = std::numbers::pi;
    std::printf("   n      h        CG iterations   max error   ratio\n");
    real previous = 0.0;
    std::vector<double> slice_x, slice_phi;
    for (idx n : {9, 17, 33}) {
        const real h = 1.0 / static_cast<real>(n - 1);
        const int m = static_cast<int>(n);
        const auto spacing = static_cast<float>(h);
        scalar_field_3d exact(m, m, m, spacing), source(m, m, m, spacing), phi(m, m, m, spacing);
        for (idx i = 0; i < n; ++i) {
            for (idx j = 0; j < n; ++j) {
                for (idx k = 0; k < n; ++k) {
                    const real x = i * h, y = j * h, z = k * h;
                    const real yz = std::sin(pi * y) * std::sin(pi * z);
                    exact(i, j, k) = std::exp(x) * std::sin(pi * x) * yz;
                    // Laplacian of the above, worked out by hand.
                    source(i, j, k) = std::exp(x) * yz *
                                      (((1.0 - (3.0 * pi * pi)) * std::sin(pi * x)) +
                                       (2.0 * pi * std::cos(pi * x)));
                }
            }
        }
        const solver_result result = field_solver::solve_poisson(phi, source, 1e-12, 2000);
        real error = 0.0;
        for (idx s = 0; s < phi.size(); ++s) {
            error = std::max(error, std::abs(phi.as_vec()[s] - exact.as_vec()[s]));
        }
        std::printf("%4zu  %7.4f  %10zu        %.2e   %s\n", static_cast<std::size_t>(n), h,
                    static_cast<std::size_t>(result.iterations), error,
                    previous > 0.0 ? std::to_string(previous / error).substr(0, 4).c_str() : "");
        previous = error;
        if (n == 33) {
            for (idx i = 0; i < n; ++i) {
                slice_x.push_back(i * h);
                slice_phi.push_back(phi(i, n / 2, n / 2));
            }
        }
    }

    if (plot) {
        plt::plot(slice_x, slice_phi, "phi(x, 1/2, 1/2)", "lines");
        plt::title("06 Poisson solution, centre slice");
        plt::show();
    }
}
