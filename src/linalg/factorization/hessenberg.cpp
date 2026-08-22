/// @file src/linalg/factorization/hessenberg.cpp
/// @brief High-performance cache-contiguous Householder upper Hessenberg decomposition.
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
    std::vector<double> w(n);
    double *H_raw = H_.data();
    double *Q_raw = Q_.data();

    // Eliminate below subdiagonal column by column
    for (idx k = 0; k < n - 2; ++k) {
        const idx m = n - 1 - k; // length of subvector to reflect

        // Compute 2-norm of H(k+1:n-1, k)
        double norm_sq = 0.0;
        for (idx i = k + 1; i < n; ++i) {
            const double val = H_raw[(i * n) + k];
            norm_sq += val * val;
        }
        const double norm_x = std::sqrt(norm_sq);
        if (norm_x < 1e-15) {
            continue;
        }

        const double x0 = H_raw[((k + 1) * n) + k];
        const double sign = (x0 >= 0.0) ? 1.0 : -1.0;
        const double mu = x0 + sign * norm_x;

        // Build Householder reflector v
        v[0] = 1.0;
        double v_norm_sq = 1.0;
        for (idx i = 1; i < m; ++i) {
            v[i] = H_raw[((k + 1 + i) * n) + k] / mu;
            v_norm_sq += v[i] * v[i];
        }
        const double beta = 2.0 / v_norm_sq;

        // 1. Left multiplication: H <- (I - beta * v * v^T) * H
        // Cache-friendly contiguous row accumulation
        for (idx j = k; j < n; ++j) {
            w[j] = 0.0;
        }
        for (idx i = 0; i < m; ++i) {
            const double vi = v[i];
            const double *row_ptr = &H_raw[(k + 1 + i) * n];
            for (idx j = k; j < n; ++j) {
                w[j] += vi * row_ptr[j];
            }
        }
        for (idx j = k; j < n; ++j) {
            w[j] *= beta;
        }
        for (idx i = 0; i < m; ++i) {
            const double vi = v[i];
            double *row_ptr = &H_raw[(k + 1 + i) * n];
            for (idx j = k; j < n; ++j) {
                row_ptr[j] -= vi * w[j];
            }
        }

        // 2. Right multiplication: H <- H * (I - beta * v * v^T)
        // Stride-1 contiguous column dot-products
        for (idx i = 0; i < n; ++i) {
            double *row_ptr = &H_raw[i * n];
            double dot = 0.0;
            for (idx j = 0; j < m; ++j) {
                dot += row_ptr[k + 1 + j] * v[j];
            }
            const double factor = beta * dot;
            for (idx j = 0; j < m; ++j) {
                row_ptr[k + 1 + j] -= factor * v[j];
            }
        }

        // 3. Accumulate into Q: Q <- Q * (I - beta * v * v^T)
        // Stride-1 contiguous updates
        for (idx i = 0; i < n; ++i) {
            double *row_ptr = &Q_raw[i * n];
            double dot = 0.0;
            for (idx j = 0; j < m; ++j) {
                dot += row_ptr[k + 1 + j] * v[j];
            }
            const double factor = beta * dot;
            for (idx j = 0; j < m; ++j) {
                row_ptr[k + 1 + j] -= factor * v[j];
            }
        }

        // Set strictly zero entries below subdiagonal
        for (idx i = k + 2; i < n; ++i) {
            H_raw[(i * n) + k] = 0.0;
        }
    }
}

} // namespace num
