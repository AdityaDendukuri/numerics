/// @file src/linalg/factorization/hessenberg.cpp
/// @brief Implementation of Householder upper Hessenberg decomposition.
#include "linalg/factorization/hessenberg.hpp"
#include <cmath>
#include <stdexcept>
#include <vector>

namespace num {

HessenbergDecomposition::HessenbergDecomposition(const Matrix &A) : H_(A), Q_(A.rows(), A.cols(), 0.0) {
    debug::check_dim(A.rows(), A.cols(), "HessenbergDecomposition matrix must be square");
    debug::check_non_empty(A.rows(), "HessenbergDecomposition matrix");

    const idx n = A.rows();
    // Initialize Q as the identity matrix
    for (idx i = 0; i < n; ++i) {
        Q_(i, i) = 1.0;
    }

    if (n <= 2) {
        return;
    }

    std::vector<double> v(n);

    // Eliminate below subdiagonal column by column
    for (idx k = 0; k < n - 2; ++k) {
        const idx m = n - 1 - k; // length of subvector to reflect

        // Compute 2-norm of H(k+1:n-1, k)
        double norm_sq = 0.0;
        for (idx i = k + 1; i < n; ++i) {
            const double val = H_(i, k);
            norm_sq += val * val;
        }
        const double norm_x = std::sqrt(norm_sq);
        if (norm_x < 1e-15) {
            continue;
        }

        const double x0 = H_(k + 1, k);
        const double sign = (x0 >= 0.0) ? 1.0 : -1.0;
        const double mu = x0 + sign * norm_x;

        // Build Householder reflector v
        v[0] = 1.0;
        double v_norm_sq = 1.0;
        for (idx i = 1; i < m; ++i) {
            v[i] = H_(k + 1 + i, k) / mu;
            v_norm_sq += v[i] * v[i];
        }
        const double beta = 2.0 / v_norm_sq;

        // 1. Left multiplication: H <- (I - beta * v * v^T) * H
        // Applies to rows k+1:n-1 and columns k:n-1
        for (idx j = k; j < n; ++j) {
            double dot = 0.0;
            for (idx i = 0; i < m; ++i) {
                dot += v[i] * H_(k + 1 + i, j);
            }
            const double factor = beta * dot;
            for (idx i = 0; i < m; ++i) {
                H_(k + 1 + i, j) -= factor * v[i];
            }
        }

        // 2. Right multiplication: H <- H * (I - beta * v * v^T)
        // Applies to rows 0:n-1 and columns k+1:n-1
        for (idx i = 0; i < n; ++i) {
            double dot = 0.0;
            for (idx j = 0; j < m; ++j) {
                dot += H_(i, k + 1 + j) * v[j];
            }
            const double factor = beta * dot;
            for (idx j = 0; j < m; ++j) {
                H_(i, k + 1 + j) -= factor * v[j];
            }
        }

        // 3. Accumulate into Q: Q <- Q * (I - beta * v * v^T)
        // Applies to rows 0:n-1 and columns k+1:n-1
        for (idx i = 0; i < n; ++i) {
            double dot = 0.0;
            for (idx j = 0; j < m; ++j) {
                dot += Q_(i, k + 1 + j) * v[j];
            }
            const double factor = beta * dot;
            for (idx j = 0; j < m; ++j) {
                Q_(i, k + 1 + j) -= factor * v[j];
            }
        }

        // Set strictly zero entries below subdiagonal
        for (idx i = k + 2; i < n; ++i) {
            H_(i, k) = 0.0;
        }
    }
}

} // namespace num
