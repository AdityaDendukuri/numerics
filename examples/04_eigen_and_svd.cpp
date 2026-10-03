/// @file 04_eigen_and_svd.cpp
/// @brief Symmetric eigendecomposition, Lanczos, power iteration and the SVD.
///
/// `eig_sym` returns every eigenpair of a matrix that claims the self-adjoint law. `lanczos` finds
/// the k largest eigenvalues from matrix-vector products alone, and `power_iteration` finds the
/// dominant one. `svd` returns the singular values of any matrix. The 1D Laplacian has eigenvalues
/// \f$2 - 2\cos(k\pi/(n+1))\f$, so the output can be checked against them.
#include <cmath>
#include <cstdio>
#include <numbers>
#include <numerics.hpp>
#include <string_view>
#include <vector>

using namespace num;

int main(int argc, char **argv) {
    const bool plot = argc > 1 && std::string_view(argv[1]) == "--plot";
    constexpr idx n = 100;
    mat<real> L(n, n, 0.0);
    for (idx i = 0; i < n; ++i) {
        L(i, i) = 2.0;
        if (i + 1 < n) {
            L(i, i + 1) = L(i + 1, i) = -1.0;
        }
    }
    auto exact = [](idx k) {
        return 2.0 - (2.0 * std::cos(static_cast<real>(k) * std::numbers::pi / (n + 1)));
    };

    const eigen_result all = eig_sym(assume_symmetric(L));
    std::printf("eig_sym     smallest %.8f  largest %.8f  (exact %.8f, %.8f)\n", all.values[0],
                all.values[n - 1], exact(1), exact(n));

    // Lanczos stops after max_steps (default min(3k, n)) and reports whether the Ritz values
    // met the tolerance. The largest eigenvalues of L are clustered, so the default is too few.
    for (idx max_steps : {0, 100}) {
        const lanczos_result top = lanczos(assume_symmetric(L), 3, 1e-10, max_steps);
        std::printf("lanczos     3 largest %.8f %.8f %.8f  %3zu steps, converged %d\n",
                    top.ritz_values[2], top.ritz_values[1], top.ritz_values[0],
                    static_cast<std::size_t>(top.steps), top.converged);
    }
    std::printf("            exact     %.8f %.8f %.8f\n", exact(n), exact(n - 1), exact(n - 2));

    const power_result dominant = power_iteration(L, 1e-10, 20000);
    std::printf("power       dominant %.8f  in %zu iterations\n", dominant.eigenvalue,
                static_cast<std::size_t>(dominant.iterations));

    // The singular values of a 4 x 3 matrix.
    mat<real> B(4, 3, 0.0);
    for (idx i = 0; i < 4; ++i) {
        for (idx j = 0; j < 3; ++j) {
            B(i, j) = static_cast<real>(i + 1) / static_cast<real>(j + 1 + i);
        }
    }
    const svd_result result = svd(B);
    std::printf("svd         singular values %.6f %.6f %.6f\n", result.S[0], result.S[1], result.S[2]);

    if (plot) {
        std::vector<double> k, numeric, analytic;
        for (idx i = 0; i < n; ++i) {
            k.push_back(static_cast<double>(i + 1));
            numeric.push_back(all.values[i]);
            analytic.push_back(exact(i + 1));
        }
        plt::plot(k, numeric, "eig_sym", "points");
        plt::plot(k, analytic, "2 - 2 cos(k pi / (n+1))", "lines");
        plt::title("04 Spectrum of the 1D Laplacian");
        plt::show();
    }
}
