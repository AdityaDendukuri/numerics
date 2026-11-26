/// @file linear/matrix_utils.hpp
/// @brief Dense matrix construction and diagonal utilities.
#pragma once

#include "core/types.hpp"
#include "container/concepts.hpp"
#include "container/matrix.hpp"
#include <algorithm>
#include <span>
#include <stdexcept>
#include <vector>

namespace num {

/// @brief Construct a unit vector: \f$e_i \in \mathbb{R}^n\f$.
[[nodiscard]] inline vec<real> unit_vector(idx size, idx index) {
    if (index >= size) {
        throw std::out_of_range("unit_vector: index out of range");
    }
    vec<real> result(size, 0.0);
    result[index] = 1.0;
    return result;
}

/// @brief Construct the identity matrix: \f$I_n\f$.
[[nodiscard]] inline mat<real> identity(idx size) {
    mat<real> result(size, size, 0.0);
    for (idx index = 0; index < size; ++index) {
        result(index, index) = 1.0;
    }
    return result;
}

/// @brief Construct a `rows x cols` matrix of zeros.
[[nodiscard]] inline mat<real> zeros(idx rows, idx cols) {
    return mat<real>(rows, cols, 0.0);
}

/// @brief Construct a vector of `size` zeros.
[[nodiscard]] inline vec<real> zeros(idx size) {
    return vec<real>(size, 0.0);
}

/// @brief Construct a `rows x cols` matrix of ones.
[[nodiscard]] inline mat<real> ones(idx rows, idx cols) {
    return mat<real>(rows, cols, 1.0);
}

/// @brief Construct a vector of `size` ones.
[[nodiscard]] inline vec<real> ones(idx size) {
    return vec<real>(size, 1.0);
}

/// @brief Construct the identity matrix: \f$I_n\f$.
[[nodiscard]] inline mat<real> eye(idx size) {
    return identity(size);
}

/// @brief Construct a rectangular identity matrix: \f$a_{ij} = \delta_{ij}\f$, `rows x cols`.
[[nodiscard]] inline mat<real> eye(idx rows, idx cols) {
    mat<real> result(rows, cols, 0.0);
    const idx k = std::min(rows, cols);
    for (idx i = 0; i < k; ++i) {
        result(i, i) = 1.0;
    }
    return result;
}

/// @brief Sum the entries of a matrix: \f$\sum_{ij} a_{ij}\f$.
[[nodiscard]] inline real accu(const mat<real> &A) {
    real sum = 0.0;
    for (idx i = 0; i < A.size(); ++i) sum += A.data()[i];
    return sum;
}

/// @brief Sum the entries of a vector: \f$\sum_i v_i\f$.
[[nodiscard]] inline real accu(const vec<real> &v) {
    real sum = 0.0;
    for (idx i = 0; i < v.size(); ++i) sum += v[i];
    return sum;
}

/// @brief Construct columns of the identity matrix: \f$[e_f, \ldots, e_{f + c - 1}]\f$.
[[nodiscard]] inline mat<real> identity_columns(idx size, idx first, idx count) {
    if (first > size || count > size - first) {
        throw std::out_of_range("identity_columns: column range out of bounds");
    }
    mat<real> result(size, count, 0.0);
    for (idx column = 0; column < count; ++column) {
        result(first + column, column) = 1.0;
    }
    return result;
}

/// @brief Return the main diagonal: \f$\mathrm{diag}(A)\f$.
template <class M>
requires repr::dense_row_major<M>
[[nodiscard]] inline vec<real> diagonal(const M &matrix) {
    const idx size = std::min(matrix.rows(), matrix.cols());
    vec<real> result(size, 0.0);
    for (idx index = 0; index < size; ++index) {
        result[index] = matrix(index, index);
    }
    return result;
}

/// @brief Construct a diagonal matrix: \f$\mathrm{diag}(d)\f$.
[[nodiscard]] inline mat<real> diagonal_matrix(view<const real> values) {
    mat<real> result(values.size(), values.size(), 0.0);
    for (idx index = 0; index < values.size(); ++index) {
        result(index, index) = values[index];
    }
    return result;
}

/// @brief Transpose a matrix: \f$A^T\f$.
template <class M>
requires repr::dense_row_major<M>
[[nodiscard]] inline mat<real> transpose(const M &matrix) {
    mat<real> result(matrix.cols(), matrix.rows(), 0.0);
    for (idx row = 0; row < matrix.rows(); ++row) {
        for (idx column = 0; column < matrix.cols(); ++column) {
            result(column, row) = matrix(row, column);
        }
    }
    return result;
}

/// @brief Set the main diagonal: \f$a_{ii} \leftarrow d_i\f$.
inline void set_diagonal(mat<real> &matrix, view<const real> values) {
    if (values.size() != std::min(matrix.rows(), matrix.cols())) {
        throw std::invalid_argument("set_diagonal: diagonal size mismatch");
    }
    for (idx index = 0; index < values.size(); ++index) {
        matrix(index, index) = values[index];
    }
}

/// @brief Multiply a vector entry by entry by weights: \f$v_i \leftarrow w_i v_i\f$.
inline void scale_elements(vec<real> &vector, view<const real> weights) {
    if (vector.size() != weights.size()) {
        throw std::invalid_argument("scale_elements: vector sizes must match");
    }
    for (idx index = 0; index < vector.size(); ++index) {
        vector[index] *= weights[index];
    }
}

/// @brief Divide a vector entry by entry: \f$v_i \leftarrow v_i / d_i\f$.
inline void divide_elements(vec<real> &vector, view<const real> divisors) {
    if (vector.size() != divisors.size()) {
        throw std::invalid_argument("divide_elements: vector sizes must match");
    }
    for (idx index = 0; index < vector.size(); ++index) {
        vector[index] /= divisors[index];
    }
}

/// @brief Scale the rows of a matrix: \f$A \leftarrow \mathrm{diag}(w) A\f$.
inline void scale_rows(mat<real> &matrix, view<const real> weights) {
    if (matrix.rows() != weights.size()) {
        throw std::invalid_argument("scale_rows: weight count must match matrix rows");
    }
    for (idx row = 0; row < matrix.rows(); ++row) {
        for (idx column = 0; column < matrix.cols(); ++column) {
            matrix(row, column) *= weights[row];
        }
    }
}

/// @brief Divide the rows of a matrix: \f$A \leftarrow \mathrm{diag}(d)^{-1} A\f$.
inline void divide_rows(mat<real> &matrix, view<const real> divisors) {
    if (matrix.rows() != divisors.size()) {
        throw std::invalid_argument("divide_rows: divisor count must match matrix rows");
    }
    for (idx row = 0; row < matrix.rows(); ++row) {
        for (idx column = 0; column < matrix.cols(); ++column) {
            matrix(row, column) /= divisors[row];
        }
    }
}

template <typename T>
/// @brief Write values at given indices, or add them:
/// \f$y_{I_k} \leftarrow x_k\f$ or \f$y_{I_k} \leftarrow y_{I_k} + x_k\f$.
void scatter(view<const T> values, view<const idx> indices, view<T> output,
             bool add = false) {
    if (values.size() != indices.size()) {
        throw std::invalid_argument("scatter: values and indices must have the same size");
    }
    for (idx position = 0; position < indices.size(); ++position) {
        if (indices[position] >= output.size()) {
            throw std::out_of_range("scatter: index out of range");
        }
        if (add) {
            output[indices[position]] += values[position];
        } else {
            output[indices[position]] = values[position];
        }
    }
}

template <typename T>
/// @brief Read values at given indices: \f$y_k = x_{I_k}\f$.
[[nodiscard]] array<T> gather(view<const T> input, view<const idx> indices) {
    array<T> result;
    result.reserve(indices.size());
    for (idx index : indices) {
        if (index >= input.size()) {
            throw std::out_of_range("gather: index out of range");
        }
        result.push_back(input[index]);
    }
    return result;
}

} // namespace num
