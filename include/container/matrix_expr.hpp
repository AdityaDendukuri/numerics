/// @file container/matrix_expr.hpp
/// @brief Value-returning dense arithmetic, and opt-in operators built on it.
///
/// Each function returns its result and checks conformance. The operators sit in `num::ops`,
/// so they need `using namespace num::ops;`. Prefer the out-parameter forms in hot loops.
#pragma once

#include "container/matrix.hpp"
#include "container/matrix_ops.hpp"
#include "container/vector.hpp"
#include "container/vector_ops.hpp"
#include <stdexcept>

namespace num {

/// Return A*B. Allocates; prefer the out-param form in hot loops.
[[nodiscard]] inline mat<real> matmul(const mat<real> &A, const mat<real> &B) {
    if (A.cols() != B.rows()) {
        throw std::invalid_argument("matmul: inner dimensions do not agree");
    }
    mat<real> C(A.rows(), B.cols(), 0.0);
    matmul(A, B, C);
    return C;
}

/// Return A*x. Allocates; prefer the out-param form in hot loops.
[[nodiscard]] inline vec<real> matvec(const mat<real> &A, const vec<real> &x) {
    if (A.cols() != x.size()) {
        throw std::invalid_argument("matvec: matrix columns do not match vector size");
    }
    vec<real> y(A.rows(), 0.0);
    matvec(A, x, y);
    return y;
}

/// Return A+B. Allocates; prefer the out-param form in hot loops.
[[nodiscard]] inline mat<real> add(const mat<real> &A, const mat<real> &B) {
    if (A.rows() != B.rows() || A.cols() != B.cols()) {
        throw std::invalid_argument("add: matrix shapes do not agree");
    }
    mat<real> C(A.rows(), A.cols(), 0.0);
    matadd(1.0, A, 1.0, B, C);
    return C;
}

/// Return A-B. Allocates; prefer the out-param form in hot loops.
[[nodiscard]] inline mat<real> sub(const mat<real> &A, const mat<real> &B) {
    if (A.rows() != B.rows() || A.cols() != B.cols()) {
        throw std::invalid_argument("sub: matrix shapes do not agree");
    }
    mat<real> C(A.rows(), A.cols(), 0.0);
    matadd(1.0, A, -1.0, B, C);
    return C;
}

/// Return x+y. Allocates; prefer the out-param form in hot loops.
[[nodiscard]] inline vec<real> add(const vec<real> &x, const vec<real> &y) {
    if (x.size() != y.size()) {
        throw std::invalid_argument("add: vector sizes do not agree");
    }
    vec<real> z(x.size(), 0.0);
    add(x, y, z);
    return z;
}

/// Return x-y. Allocates; prefer the out-param form in hot loops.
[[nodiscard]] inline vec<real> sub(const vec<real> &x, const vec<real> &y) {
    if (x.size() != y.size()) {
        throw std::invalid_argument("sub: vector sizes do not agree");
    }
    vec<real> z(x);
    axpy(-1.0, y, z);
    return z;
}

/// Return alpha*A, matching the value-returning `scaled` for sparse matrices.
/// Allocates; prefer in-place scaling in hot loops.
[[nodiscard]] inline mat<real> scaled(const mat<real> &A, real alpha) {
    mat<real> result(A.rows(), A.cols(), 0.0);
    matadd(alpha, A, 0.0, A, result);
    return result;
}

/// Return alpha*x for vectors.
[[nodiscard]] inline vec<real> scaled(const vec<real> &x, real alpha) {
    vec<real> result = x;
    scale(result, alpha);
    return result;
}

/// Opt-in operator spellings for the functions above.
///
/// These are deliberately not in `num`, so that no translation unit acquires
/// them by including a header.  Write `using namespace num::ops;` to enable
/// them where expression syntax makes a formula clearer.
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
