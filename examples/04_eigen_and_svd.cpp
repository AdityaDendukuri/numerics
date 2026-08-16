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

    // 1. Full Symmetric Eigendecomposition
    auto eig_res = eig_sym(A, 1e-12, 100, Backend::lapack);
    std::cout << "Eigenvalues: [" << eig_res.values[0] << ", " 
              << eig_res.values[1] << ", " << eig_res.values[2] << "]\n";

    // 2. SVD
    auto svd_res = svd(A, Backend::lapack);
    std::cout << "Singular Values: [" << svd_res.S[0] << ", " 
              << svd_res.S[1] << ", " << svd_res.S[2] << "]\n";

    // Plot spectrum with high-res ASCII dimensions (140 cols x 35 rows)
    std::vector<double> mode_idx{1.0, 2.0, 3.0};
    std::vector<double> eigs{eig_res.values[0], eig_res.values[1], eig_res.values[2]};
    std::vector<double> svs{svd_res.S[0], svd_res.S[1], svd_res.S[2]};

    plt::plot(mode_idx, eigs, "Eigenvalues lambda_k", "linespoints");
    plt::plot(mode_idx, svs, "Singular Values sigma_k", "linespoints");
    plt::title("04 Eigensolvers & SVD: Spectral Spectrum Comparison");
    plt::xlabel("Mode Index k");
    plt::ylabel("Spectral Magnitude");
    plt::legend();
    plt::show_dumb(140, 35);

    return 0;
}
