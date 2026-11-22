/// @file linear/sparse/bilinear.hpp
/// @brief Sparse bilinear contractions without dense intermediate matrices.
#pragma once

#include "container/matrix.hpp"
#include "linear/sparse/sparse.hpp"

#include <stdexcept>

namespace num {

/// y^T A x for vectors and a real CSR matrix; no conjugation is applied.
template <typename T>
inline T sparse_bilinear(const spmat &A, const vec<T> &y, const vec<T> &x) {
    if (y.size() != A.n_rows() || x.size() != A.n_cols()) {
        throw std::invalid_argument("sparse_bilinear: dimension mismatch");
    }
    T result{};
    for (idx row = 0; row < A.n_rows(); ++row) {
        for (idx entry = A.row_ptr()[row]; entry < A.row_ptr()[row + 1]; ++entry) {
            result += A.values()[entry] * y[row] * x[A.col_idx()[entry]];
        }
    }
    return result;
}

/// trace(L^T A R), where L and R are row-major dense matrices with equal column counts.
inline real sparse_bilinear_trace(const spmat &A, const mat<real> &L, const mat<real> &R) {
    if (L.rows() != A.n_rows() || R.rows() != A.n_cols() || L.cols() != R.cols()) {
        throw std::invalid_argument("sparse_bilinear_trace: dimension mismatch");
    }
    const idx width = L.cols();
    if (width == 0) {
        return 0.0;
    }
    real result = 0.0;
    for (idx row = 0; row < A.n_rows(); ++row) {
        const real *left = L.data() + row * width;
        for (idx entry = A.row_ptr()[row]; entry < A.row_ptr()[row + 1]; ++entry) {
            const real *right = R.data() + A.col_idx()[entry] * width;
            real inner = 0.0;
            for (idx column = 0; column < width; ++column) {
                inner += left[column] * right[column];
            }
            result += A.values()[entry] * inner;
        }
    }
    return result;
}

} // namespace num
