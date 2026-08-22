/// @file 02_iterative_krylov_solvers.cpp
/// @brief CG, PCG, MINRES, GMRES, and Matrix-Free Operators.
#include <iostream>
#include <numerics.hpp>
#include <vector>

int main() {
    using namespace num;

    idx n = 100;
    std::vector<idx> rows, cols;
    std::vector<real> vals;
    for (idx i = 0; i < n; ++i) {
        rows.push_back(i);
        cols.push_back(i);
        vals.push_back(4.0);
        if (i > 0) {
            rows.push_back(i);
            cols.push_back(i - 1);
            vals.push_back(-1.0);
        }
        if (i + 1 < n) {
            rows.push_back(i);
            cols.push_back(i + 1);
            vals.push_back(-1.0);
        }
    }
    SparseMatrix A = SparseMatrix::from_triplets(n, n, rows, cols, vals);

    Vector b(n, 1.0);
    Vector x0(n, 0.0);

    // 1. Conjugate Gradient (CG)
    operators::SparseOp aop(A);
    auto spd_a = operators::assume_spd(aop);
    auto cg_res = cg(spd_a, b, x0, 1e-8, 500);
    std::cout << "CG Converged: " << (cg_res.converged ? "YES" : "NO") << " in "
              << cg_res.iterations << " iters. Residual = " << cg_res.residual << "\n";

    // 2. GMRES Solver
    auto gmres_res = gmres(aop, b, x0, 1e-8, 500, 30);
    std::cout << "GMRES Converged: " << (gmres_res.converged ? "YES" : "NO") << " in "
              << gmres_res.iterations << " iters. Residual = " << gmres_res.residual << "\n";

    // Plot solution vector x over grid (140x35)
    std::vector<double> grid, sol;
    for (idx i = 0; i < n; ++i) {
        grid.push_back(static_cast<double>(i));
        sol.push_back(x0[i]);
    }
    plt::plot(grid, sol, "x_cg", "lines");
    plt::title("02 Krylov Iterative Solvers: Solution Vector x");
    plt::xlabel("Grid Node i");
    plt::ylabel("Solution x_i");
    plt::show_dumb(140, 35);

    return 0;
}
