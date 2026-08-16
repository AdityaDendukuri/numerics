/// @file 04_eigen_and_svd.cpp
/// @brief Jacobi, LAPACK dsyevd, Lanczos top-k eigensolvers, and SVD.
#include <numerics.hpp>
#include <iostream>

int main() {
    using namespace num;

    Matrix A(3, 3, 0.0);
    A(0,0) = 2.0; A(0,1) = -1.0; A(0,2) = 0.0;
    A(1,0) = -1.0; A(1,1) = 2.0; A(1,2) = -1.0;
    A(2,0) = 0.0; A(2,1) = -1.0; A(2,2) = 2.0;

    // 1. Full Symmetric Eigendecomposition (Jacobi / LAPACK dsyevd)
    auto eig_res = eig_sym(A, 1e-12, 100, Backend::lapack);
    std::cout << "Eigenvalues: [" << eig_res.values[0] << ", " 
              << eig_res.values[1] << ", " << eig_res.values[2] << "]\n";

    // 2. Lanczos Eigensolver (Top-K Eigenvalues)
    auto lanczos_res = lanczos(A, 2, 20);
    std::cout << "Lanczos Top-2 Eigenvalues: [" << lanczos_res.ritz_values[0] 
              << ", " << lanczos_res.ritz_values[1] << "]\n";

    // 3. Full Singular Value Decomposition (SVD)
    auto svd_res = svd(A, Backend::lapack);
    std::cout << "Singular Values: [" << svd_res.S[0] << ", " 
              << svd_res.S[1] << ", " << svd_res.S[2] << "]\n";

    // 4. Fast Truncated SVD
    auto rsvd_res = svd_truncated(A, 2);
    std::cout << "Truncated Top-2 Singular Values: [" << rsvd_res.S[0] << ", " << rsvd_res.S[1] << "]\n";

    return 0;
}
