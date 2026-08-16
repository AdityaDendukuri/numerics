/// @file 02_iterative_krylov_solvers.cpp
/// @brief CG, PCG, MINRES, GMRES, and Matrix-Free Operators.
#include <numerics.hpp>
#include <iostream>
#include <vector>

int main() {
    using namespace num;

    idx n = 100;
    std::vector<idx> rows, cols;
    std::vector<real> vals;
    for (idx i = 0; i < n; ++i) {
        rows.push_back(i); cols.push_back(i); vals.push_back(4.0);
        if (i > 0) { rows.push_back(i); cols.push_back(i - 1); vals.push_back(-1.0); }
        if (i + 1 < n) { rows.push_back(i); cols.push_back(i + 1); vals.push_back(-1.0); }
    }
    SparseMatrix A = SparseMatrix::from_triplets(n, n, rows, cols, vals);

    Vector b(n, 1.0);
    Vector x0(n, 0.0);

    // 1. Conjugate Gradient (CG) with SPD operator tag
    operators::SparseOp Aop(A);
    auto spd_A = operators::assume_spd(Aop);
    auto cg_res = cg(spd_A, b, x0, 1e-8, 500);
    std::cout << "CG Converged: " << (cg_res.converged ? "YES" : "NO") 
              << " in " << cg_res.iterations << " iters. Residual = " << cg_res.residual << "\n";

    // 2. GMRES Solver for non-symmetric / general sparse systems
    auto gmres_res = gmres(Aop, b, x0, 1e-8, 500, 30);
    std::cout << "GMRES Converged: " << (gmres_res.converged ? "YES" : "NO") 
              << " in " << gmres_res.iterations << " iters. Residual = " << gmres_res.residual << "\n";

    // 3. Matrix-Free Custom Operator Solve
    auto free_op = operators::make_op(
        [n](const Vector& v, Vector& Av) {
            for (idx i = 0; i < n; ++i) {
                Av[i] = 4.0 * v[i];
                if (i > 0) Av[i] -= v[i - 1];
                if (i + 1 < n) Av[i] -= v[i + 1];
            }
        }, n, n);
    auto spd_free = operators::assume_spd(free_op);
    auto free_res = cg(spd_free, b, x0, 1e-8, 500);
    std::cout << "Matrix-Free CG Converged in " << free_res.iterations << " iters.\n";

    return 0;
}
