/// @file container/matrix_expr.hpp
/// @brief Value-returning dense arithmetic, with opt-in operators in `num::ops`.
#pragma once

#include "container/matrix.hpp"
#include "container/matrix_ops.hpp"
#include "container/vector.hpp"
#include "container/vector_ops.hpp"
#include <stdexcept>

namespace num {

/// @brief Multiply two matrices: \f$A B\f$. Allocates.
[[nodiscard]] inline mat<real> matmul(const mat<real> &A, const mat<real> &B) {
    if (A.cols() != B.rows()) {
        throw std::invalid_argument("matmul: inner dimensions do not agree");
    }
    mat<real> C(A.rows(), B.cols(), 0.0);
    matmul(A, B, C);
    return C;
}

/// @brief Multiply a matrix and a vector: \f$A x\f$. Allocates.
[[nodiscard]] inline vec<real> matvec(const mat<real> &A, const vec<real> &x) {
    if (A.cols() != x.size()) {
        throw std::invalid_argument("matvec: matrix columns do not match vector size");
    }
    vec<real> y(A.rows(), 0.0);
    matvec(A, x, y);
    return y;
}

/// @brief Add two matrices: \f$A + B\f$. Allocates.
[[nodiscard]] inline mat<real> add(const mat<real> &A, const mat<real> &B) {
    if (A.rows() != B.rows() || A.cols() != B.cols()) {
        throw std::invalid_argument("add: matrix shapes do not agree");
    }
    mat<real> C(A.rows(), A.cols(), 0.0);
    matadd(1.0, A, 1.0, B, C);
    return C;
}

/// @brief Subtract two matrices: \f$A - B\f$. Allocates.
[[nodiscard]] inline mat<real> sub(const mat<real> &A, const mat<real> &B) {
    if (A.rows() != B.rows() || A.cols() != B.cols()) {
        throw std::invalid_argument("sub: matrix shapes do not agree");
    }
    mat<real> C(A.rows(), A.cols(), 0.0);
    matadd(1.0, A, -1.0, B, C);
    return C;
}

/// @brief Add two vectors: \f$x + y\f$. Allocates.
[[nodiscard]] inline vec<real> add(const vec<real> &x, const vec<real> &y) {
    if (x.size() != y.size()) {
        throw std::invalid_argument("add: vector sizes do not agree");
    }
    vec<real> z(x.size(), 0.0);
    add(x, y, z);
    return z;
}

/// @brief Subtract two vectors: \f$x - y\f$. Allocates.
[[nodiscard]] inline vec<real> sub(const vec<real> &x, const vec<real> &y) {
    if (x.size() != y.size()) {
        throw std::invalid_argument("sub: vector sizes do not agree");
    }
    vec<real> z(x);
    axpy(-1.0, y, z);
    return z;
}

/// @brief Scale a matrix: \f$\alpha A\f$. Allocates.
[[nodiscard]] inline mat<real> scaled(const mat<real> &A, real alpha) {
    mat<real> result(A.rows(), A.cols(), 0.0);
    matadd(alpha, A, 0.0, A, result);
    return result;
}

/// @brief Scale a vector: \f$\alpha x\f$. Allocates.
[[nodiscard]] inline vec<real> scaled(const vec<real> &x, real alpha) {
    vec<real> result = x;
    scale(result, alpha);
    return result;
}

/// @brief Operators `*`, `+` and `-` for the functions above; enable with `using namespace
/// num::ops;`.
namespace ops {

[[nodiscard]] inline mat<real> operator*(const mat<real> &A, const mat<real> &B) { return matmul(A, B); }
[[nodiscard]] inline vec<real> operator*(const mat<real> &A, const vec<real> &x) { return matvec(A, x); }
[[nodiscard]] inline mat<real> operator+(const mat<real> &A, const mat<real> &B) { return add(A, B); }
[[nodiscard]] inline mat<real> operator-(const mat<real> &A, const mat<real> &B) { return sub(A, B); }
[[nodiscard]] inline vec<real> operator+(const vec<real> &x, const vec<real> &y) { return add(x, y); }
[[nodiscard]] inline vec<real> operator-(const vec<real> &x, const vec<real> &y) { return sub(x, y); }
[[nodiscard]] inline mat<real> operator*(const mat<real> &A, real alpha) { return scaled(A, alpha); }
[[nodiscard]] inline mat<real> operator*(real alpha, const mat<real> &A) { return scaled(A, alpha); }
[[nodiscard]] inline mat<real> operator/(const mat<real> &A, real alpha) { return scaled(A, 1.0 / alpha); }
[[nodiscard]] inline vec<real> operator*(const vec<real> &x, real alpha) { return scaled(x, alpha); }
[[nodiscard]] inline vec<real> operator*(real alpha, const vec<real> &x) { return scaled(x, alpha); }
[[nodiscard]] inline vec<real> operator/(const vec<real> &x, real alpha) { return scaled(x, 1.0 / alpha); }

} // namespace ops

} // namespace num
