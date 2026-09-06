/// @file 01_direct_factorizations.cpp
/// @brief LU, QR, Cholesky, and Thomas tridiagonal direct factorizations.
#include <iostream>
#include <numerics.hpp>

int main() {
    using namespace num;

    // 1. LU Factorization (PA = LU)
    mat A(3, 3, 0.0);
    A(0, 0) = 4.0;
    A(0, 1) = 1.0;
    A(0, 2) = 0.0;
    A(1, 0) = 1.0;
    A(1, 1) = 4.0;
    A(1, 2) = 1.0;
    A(2, 0) = 0.0;
    A(2, 1) = 1.0;
    A(2, 2) = 4.0;
    vec b{5.0, 6.0, 5.0};

    auto lu_fact = lu(assume_square(A));
    vec x_lu;
    lu_solve(lu_fact, b, x_lu);
    std::cout << "LU Solve x = [" << x_lu[0] << ", " << x_lu[1] << ", " << x_lu[2] << "]\n";

    // 2. QR Factorization (A = QR)
    auto qr_fact = qr(A);
    vec x_qr;
    qr_solve(qr_fact, b, x_qr);

    // 3. Cholesky Factorization (A = L L^T) with compile-time SPD concept tagging
    auto chol_fact = cholesky(assume_spd(A));
    vec x_chol(3, 0.0);
    cholesky_solve(chol_fact, b, x_chol);
    std::cout << "Cholesky Solve x = [" << x_chol[0] << ", " << x_chol[1] << ", " << x_chol[2] << "]\n";

    // 4. Thomas tridiagonal Algorithm (O(n) solve)
    vec dl{1.0, 1.0};
    vec d{4.0, 4.0, 4.0};
    vec du{1.0, 1.0};
    vec x_thomas(3, 0.0);
    thomas(dl, d, du, b, x_thomas);

    // Terminal ASCII Plot (140x35)
    std::vector<double> idx_vec{0.0, 1.0, 2.0};
    std::vector<double> sol_vec{x_lu[0], x_lu[1], x_lu[2]};
    plt::plot(idx_vec, sol_vec, "x_solution", "linespoints");
    plt::title("01 Direct Factorizations: Solution vec x");
    plt::xlabel("Index i");
    plt::ylabel("Solution x_i");
    plt::show_dumb(140, 35);

    return 0;
}
