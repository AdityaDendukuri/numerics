/// @file 01_direct_factorizations.cpp
/// @brief LU, QR, Cholesky, and Thomas Tridiagonal direct factorizations.
#include <numerics.hpp>
#include <iostream>
#include <cassert>
#include <cmath>

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
    std::cout << "QR Solve x = [" << x_qr[0] << ", " << x_qr[1] << ", " << x_qr[2] << "]\n";

    // 3. Cholesky Factorization (A = L L^T)
    auto chol_fact = cholesky(A);
    Vector x_chol(3, 0.0);
    cholesky_solve(chol_fact, b, x_chol);
    std::cout << "Cholesky Solve x = [" << x_chol[0] << ", " << x_chol[1] << ", " << x_chol[2] << "]\n";

    // 4. Thomas Tridiagonal Algorithm (O(n) solve)
    Vector dl{1.0, 1.0};
    Vector d{4.0, 4.0, 4.0};
    Vector du{1.0, 1.0};
    Vector x_thomas(3, 0.0);
    thomas(dl, d, du, b, x_thomas);
    std::cout << "Thomas Solve x = [" << x_thomas[0] << ", " << x_thomas[1] << ", " << x_thomas[2] << "]\n";

    return 0;
}
