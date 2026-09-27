/// @file linear/matrix_properties.hpp
/// @brief Exact symmetry and definiteness checks for dense matrices, and the laws they attach.
#pragma once

#include "container/concepts.hpp"
#include "container/matrix.hpp"
#include "container/vector.hpp"
#include "kernel/factor.hpp"
#include "operator/properties.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace num {

namespace linear {

/// Maximum absolute difference between mirrored entries of a square matrix.
[[nodiscard]] inline real symmetry_error(const mat &A) {
    if (A.rows() != A.cols()) {
        throw std::invalid_argument("symmetry_error: matrix must be square");
    }
    real error = 0.0;
    for (idx row = 0; row < A.rows(); ++row) {
        for (idx column = 0; column < row; ++column) {
            error = std::max(error, std::abs(A(row, column) - A(column, row)));
        }
    }
    return error;
}

/// Maximum mirrored-entry error relative to the largest off-diagonal entry.
[[nodiscard]] inline real relative_symmetry_error(const mat &A) {
    if (A.rows() != A.cols()) {
        throw std::invalid_argument("relative_symmetry_error: matrix must be square");
    }
    real error = 0.0;
    real scale = 1.0;
    for (idx row = 0; row < A.rows(); ++row) {
        for (idx column = 0; column < row; ++column) {
            error = std::max(error, std::abs(A(row, column) - A(column, row)));
            scale = std::max(scale, std::abs(A(row, column)));
            scale = std::max(scale, std::abs(A(column, row)));
        }
    }
    return error / scale;
}

/// Test absolute entrywise symmetry using the supplied tolerance.
[[nodiscard]] inline bool is_symmetric(const mat &A, real tol = 1e-12) {
    if (A.rows() != A.cols()) {
        return false;
    }
    const idx n = A.rows();
    for (idx i = 0; i < n; ++i) {
        for (idx j = 0; j < i; ++j) {
            if (std::abs(A(i, j) - A(j, i)) > tol) {
                return false;
            }
        }
    }
    return true;
}

/// Test symmetry and positive definiteness by a Cholesky factorization, which fails exactly
/// when a pivot is not positive. The raw kernel is used because `num::cholesky` requires the
/// invariant being tested.
[[nodiscard]] inline bool is_spd(const mat &A, real tol = 1e-12) {
    if (!is_symmetric(A, tol)) {
        return false;
    }
    mat factor(A.rows(), A.cols(), 0.0);
    return kernel::cholesky(factor.data(), A.data(), A.rows());
}

/// @brief Check symmetry \f$\max_{i,j} |A_{ij} - A_{ji}| \le \mathrm{tol}\f$ exhaustively, then
/// attach it.
/// @throws std::invalid_argument If `A` is not symmetric within `tol`.
template <class Mat = mat>
[[nodiscard]] inline with_law<Mat, law::self_adjoint>
make_symmetric(Mat A, real tol = 1e-12) {
    if (!is_symmetric(A, tol)) {
        throw std::invalid_argument("make_symmetric: matrix is not symmetric");
    }
    return with_law<Mat, law::self_adjoint>(std::move(A));
}

/// @brief Check symmetric positive definiteness exhaustively by a Cholesky factorization, then
/// attach it.
/// @throws std::invalid_argument If `A` is not symmetric within `tol`, or a pivot is not positive.
template <class Mat = mat>
[[nodiscard]] inline with_law<Mat, law::spd>
make_spd(Mat A, real tol = 1e-12) {
    if (!is_spd(A, tol)) {
        throw std::invalid_argument("make_spd: matrix is not symmetric positive definite");
    }
    return with_law<Mat, law::spd>(std::move(A));
}

} // namespace linear

using linear::make_spd;
using linear::make_symmetric;

} // namespace num
