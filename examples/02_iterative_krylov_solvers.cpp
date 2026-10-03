/// @file 02_iterative_krylov_solvers.cpp
/// @brief CG, preconditioned CG, MINRES and GMRES on sparse and matrix-free operators.
///
/// Each solver states its requirement as a concept. CG and PCG need an SPD operator, MINRES needs a
/// self-adjoint one, and GMRES accepts any linear operator. A law is attached to an operator with
/// `assume_spd` or `assume_symmetric`, and an operator without the required law does not compile.
#include <cmath>
#include <cstdio>
#include <numerics.hpp>
#include <string_view>
#include <vector>

using namespace num;

namespace {

// Diffusion with a coefficient that grows along the domain, so the diagonal varies.
real coefficient(idx i) {
    return 2.01 + (0.05 * static_cast<real>(i));
}

// Tridiagonal (lower, coefficient(i), upper) as a sparse matrix.
spmat tridiagonal(idx n, real lower, real upper) {
    array<idx> rows, columns;
    array<real> values;
    for (idx i = 0; i < n; ++i) {
        rows.push_back(i);
        columns.push_back(i);
        values.push_back(coefficient(i));
        if (i > 0) {
            rows.push_back(i);
            columns.push_back(i - 1);
            values.push_back(lower);
        }
        if (i + 1 < n) {
            rows.push_back(i);
            columns.push_back(i + 1);
            values.push_back(upper);
        }
    }
    return spmat::from_triplets(n, n, rows, columns, values);
}

void report(const char *name, const solver_result &r) {
    std::printf("%-30s converged %d  iterations %4zu  residual %.2e\n", name, r.converged,
                static_cast<std::size_t>(r.iterations), r.residual);
}

} // namespace

int main(int argc, char **argv) {
    const bool plot = argc > 1 && std::string_view(argv[1]) == "--plot";
    constexpr idx n = 400;
    const vec<real> b(n, 1.0);

    // A symmetric positive definite matrix with a varying diagonal.
    const spmat A = tridiagonal(n, -1.0, -1.0);
    const operators::sparse_op op(A);

    vec<real> x(n, 0.0);
    report("CG, sparse", cg(assume_spd(op), b, x, {.tolerance = 1e-10, .max_iterations = 2000}));

    x = vec<real>(n, 0.0);
    vec<real> inverse_diagonal(n);
    for (idx i = 0; i < n; ++i) {
        inverse_diagonal[i] = 1.0 / coefficient(i);
    }
    const jacobi_preconditioner jacobi(inverse_diagonal);
    report("PCG, Jacobi",
           pcg(assume_spd(op), jacobi, b, x, {.tolerance = 1e-10, .max_iterations = 2000}));

    x = vec<real>(n, 0.0);
    report("MINRES", minres(assume_symmetric(op), b, x, {.tolerance = 1e-10, .max_iterations = 2000}));

    // A nonsymmetric matrix: convection-diffusion. Only GMRES applies.
    const spmat N = tridiagonal(n, -1.4, -0.6);
    vec<real> y(n, 0.0);
    report("GMRES(30), nonsymmetric",
           gmres(operators::sparse_op(N), b, y, {.tolerance = 1e-10, .max_iterations = 4000, .restart = 30}));

    // A matrix-free operator: the same SPD matrix as a lambda.
    const auto laplacian = operators::make_op(
        [](const vec<real> &u, vec<real> &Lu) {
            const idx m = u.size();
            for (idx i = 0; i < m; ++i) {
                Lu[i] = (coefficient(i) * u[i]) - (i > 0 ? u[i - 1] : 0.0) - (i + 1 < m ? u[i + 1] : 0.0);
            }
        },
        n);
    vec<real> u(n, 0.0);
    report("CG, matrix-free", cg(assume_spd(laplacian), b, u, {.tolerance = 1e-10, .max_iterations = 2000}));
    std::printf("max |x_sparse - x_matrix_free| = %.1e\n", [&] {
        real worst = 0.0;
        for (idx i = 0; i < n; ++i) {
            worst = std::max(worst, std::abs(x[i] - u[i]));
        }
        return worst;
    }());

    if (plot) {
        std::vector<double> grid, sol;
        for (idx i = 0; i < n; ++i) {
            grid.push_back(static_cast<double>(i));
            sol.push_back(u[i]);
        }
        plt::plot(grid, sol, "x", "lines");
        plt::title("02 CG solution");
        plt::show();
    }
}
