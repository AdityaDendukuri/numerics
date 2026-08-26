/// @file linear/matrix_utils.hpp
/// @brief Small dense matrix construction and diagonal utilities.
#pragma once

#include "container/matrix.hpp"
#include <algorithm>
#include <span>
#include <stdexcept>
#include <vector>

namespace num {

/// Construct the selected standard basis vector in R^size.
[[nodiscard]] inline Vector unit_vector(idx size, idx index) {
    if (index >= size) {
        throw std::out_of_range("unit_vector: index out of range");
    }
    Vector result(size, 0.0);
    result[index] = 1.0;
    return result;
}

/// Construct the square identity matrix of the requested size.
[[nodiscard]] inline Matrix identity(idx size) {
    Matrix result(size, size, 0.0);
    for (idx index = 0; index < size; ++index) {
        result(index, index) = 1.0;
    }
    return result;
}

/// Extract a consecutive block of columns from an implicit identity matrix.
[[nodiscard]] inline Matrix identity_columns(idx size, idx first, idx count) {
    if (first > size || count > size - first) {
        throw std::out_of_range("identity_columns: column range out of bounds");
    }
    Matrix result(size, count, 0.0);
    for (idx column = 0; column < count; ++column) {
        result(first + column, column) = 1.0;
    }
    return result;
}

/// Extract the main diagonal up to the smaller matrix dimension.
[[nodiscard]] inline Vector diagonal(const Matrix &matrix) {
    const idx size = std::min(matrix.rows(), matrix.cols());
    Vector result(size, 0.0);
    for (idx index = 0; index < size; ++index) {
        result[index] = matrix(index, index);
    }
    return result;
}

/// Construct a square matrix with the supplied main diagonal.
[[nodiscard]] inline Matrix diagonal_matrix(std::span<const real> values) {
    Matrix result(values.size(), values.size(), 0.0);
    for (idx index = 0; index < values.size(); ++index) {
        result(index, index) = values[index];
    }
    return result;
}

/// Return the transpose of a dense matrix.
[[nodiscard]] inline Matrix transpose(const Matrix &matrix) {
    Matrix result(matrix.cols(), matrix.rows(), 0.0);
    for (idx row = 0; row < matrix.rows(); ++row) {
        for (idx column = 0; column < matrix.cols(); ++column) {
            result(column, row) = matrix(row, column);
        }
    }
    return result;
}

/// Replace a matrix's full main diagonal.
inline void set_diagonal(Matrix &matrix, std::span<const real> values) {
    if (values.size() != std::min(matrix.rows(), matrix.cols())) {
        throw std::invalid_argument("set_diagonal: diagonal size mismatch");
    }
    for (idx index = 0; index < values.size(); ++index) {
        matrix(index, index) = values[index];
    }
}

/// Multiply a vector elementwise by matching weights in place.
inline void scale_elements(Vector &vector, std::span<const real> weights) {
    if (vector.size() != weights.size()) {
        throw std::invalid_argument("scale_elements: vector sizes must match");
    }
    for (idx index = 0; index < vector.size(); ++index) {
        vector[index] *= weights[index];
    }
}

/// Divide a vector elementwise by matching divisors in place.
inline void divide_elements(Vector &vector, std::span<const real> divisors) {
    if (vector.size() != divisors.size()) {
        throw std::invalid_argument("divide_elements: vector sizes must match");
    }
    for (idx index = 0; index < vector.size(); ++index) {
        vector[index] /= divisors[index];
    }
}

/// Multiply each matrix row by its corresponding weight in place.
inline void scale_rows(Matrix &matrix, std::span<const real> weights) {
    if (matrix.rows() != weights.size()) {
        throw std::invalid_argument("scale_rows: weight count must match matrix rows");
    }
    for (idx row = 0; row < matrix.rows(); ++row) {
        for (idx column = 0; column < matrix.cols(); ++column) {
            matrix(row, column) *= weights[row];
        }
    }
}

/// Divide each matrix row by its corresponding divisor in place.
inline void divide_rows(Matrix &matrix, std::span<const real> divisors) {
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
/// Write values at the supplied indices, optionally accumulating into output.
void scatter(std::span<const T> values, std::span<const idx> indices, std::span<T> output,
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
/// Copy the indexed input entries into a compact vector.
[[nodiscard]] std::vector<T> gather(std::span<const T> input, std::span<const idx> indices) {
    std::vector<T> result;
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
