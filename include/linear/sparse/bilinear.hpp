/// @file linear/sparse/bilinear.hpp
/// @brief Sparse bilinear forms without dense intermediates.
#pragma once

#include "container/matrix.hpp"
#include "linear/sparse/sparse.hpp"

#include <stdexcept>

namespace num {

/// @brief Evaluate a sparse bilinear form, without conjugation: \f$y^T A x\f$.
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

/// @brief Evaluate the trace of a product with a sparse matrix: \f$\mathrm{tr}(L^T A R)\f$.
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
