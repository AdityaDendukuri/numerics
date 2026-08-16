/// @file 01_direct_factorizations.cpp
/// @brief LU, QR, Cholesky, and Thomas Tridiagonal direct factorizations.
#include <numerics.hpp>
#include <iostream>

int main() {
    using namespace num;

    // 1. LU Factorization (PA = LU)
    Matrix A(3, 3, 0.0);
    A(0,0) = 4.0; A(0,1) = 1.0; A(0,2) = 0.0;
    A(1,0) = 1.0; A(1,1) = 4.0; A(1,2) = 1.0;
    A(2,0) = 0.0; A(2,1) = 1.0; A(2,2) = 4.0;
    Vector b{5.0, 6.0, 5.0};

    auto lu_fact = lu(A);
    Vector x_lu;
    lu_solve(lu_fact, b, x_lu);
    std::cout << "LU Solve x = [" << x_lu[0] << ", " << x_lu[1] << ", " << x_lu[2] << "]\n";

    // 2. QR Factorization (A = QR)
    auto qr_fact = qr(A);
    Vector x_qr;
    qr_solve(qr_fact, b, x_qr);

    // 3. Cholesky Factorization (A = L L^T)
    auto chol_fact = cholesky(A);
    Vector x_chol(3, 0.0);
    cholesky_solve(chol_fact, b, x_chol);

    // 4. Thomas Tridiagonal Algorithm (O(n) solve)
    Vector dl{1.0, 1.0};
    Vector d{4.0, 4.0, 4.0};
    Vector du{1.0, 1.0};
    Vector x_thomas(3, 0.0);
    thomas(dl, d, du, b, x_thomas);

    // Terminal ASCII Plot
    std::vector<double> idx_vec{0.0, 1.0, 2.0};
    std::vector<double> sol_vec{x_lu[0], x_lu[1], x_lu[2]};
    plt::plot(idx_vec, sol_vec, "x_solution", "linespoints");
    plt::title("01 Direct Factorizations: Solution Vector x");
    plt::xlabel("Index i");
    plt::ylabel("Solution x_i");
    plt::show_dumb(100, 20);

    return 0;
}
