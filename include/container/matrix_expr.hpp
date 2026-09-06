/// @file container/matrix_expr.hpp
/// @brief Value-returning dense arithmetic, and opt-in operators built on it.
///
/// Numerics' primary kernels write into caller-provided buffers so that no
/// allocation happens inside a simulation loop.  That remains the default.  The
/// functions here are the convenience tier above them: each returns its result
/// and checks conformance, which the out-parameter forms cannot do because the
/// caller has already sized the output.
///
/// Operators live in the nested `num::ops` namespace and are therefore opt-in.
/// Ordinary lookup does not find them until a translation unit asks:
///
/// ```cpp
/// using namespace num::ops;
/// const vec r = y - Z * q;
/// ```
///
/// Prefer the out-parameter forms in hot loops; prefer these where the code is
/// stating a formula and clarity is worth one temporary.
#pragma once

#include "container/matrix.hpp"
#include "container/matrix_ops.hpp"
#include "container/vector.hpp"
#include "container/vector_ops.hpp"
#include <stdexcept>

namespace num {

/// Return A*B. Allocates; prefer the out-param form in hot loops.
[[nodiscard]] inline mat matmul(const mat &A, const mat &B) {
    if (A.cols() != B.rows()) {
        throw std::invalid_argument("matmul: inner dimensions do not agree");
    }
    mat C(A.rows(), B.cols(), 0.0);
    matmul(A, B, C);
    return C;
}

/// Return A*x. Allocates; prefer the out-param form in hot loops.
[[nodiscard]] inline vec matvec(const mat &A, const vec &x) {
    if (A.cols() != x.size()) {
        throw std::invalid_argument("matvec: matrix columns do not match vector size");
    }
    vec y(A.rows(), 0.0);
    matvec(A, x, y);
    return y;
}

/// Return A+B. Allocates; prefer the out-param form in hot loops.
[[nodiscard]] inline mat add(const mat &A, const mat &B) {
    if (A.rows() != B.rows() || A.cols() != B.cols()) {
        throw std::invalid_argument("add: matrix shapes do not agree");
    }
    mat C(A.rows(), A.cols(), 0.0);
    matadd(1.0, A, 1.0, B, C);
    return C;
}

/// Return A-B. Allocates; prefer the out-param form in hot loops.
[[nodiscard]] inline mat sub(const mat &A, const mat &B) {
    if (A.rows() != B.rows() || A.cols() != B.cols()) {
        throw std::invalid_argument("sub: matrix shapes do not agree");
    }
    mat C(A.rows(), A.cols(), 0.0);
    matadd(1.0, A, -1.0, B, C);
    return C;
}

/// Return x+y. Allocates; prefer the out-param form in hot loops.
[[nodiscard]] inline vec add(const vec &x, const vec &y) {
    if (x.size() != y.size()) {
        throw std::invalid_argument("add: vector sizes do not agree");
    }
    vec z(x.size(), 0.0);
    add(x, y, z);
    return z;
}

/// Return x-y. Allocates; prefer the out-param form in hot loops.
[[nodiscard]] inline vec sub(const vec &x, const vec &y) {
    if (x.size() != y.size()) {
        throw std::invalid_argument("sub: vector sizes do not agree");
    }
    vec z(x);
    axpy(-1.0, y, z);
    return z;
}

/// Return alpha*A, matching the value-returning `scaled` for sparse matrices.
/// Allocates; prefer in-place scaling in hot loops.
[[nodiscard]] inline mat scaled(const mat &A, real alpha) {
    mat result(A.rows(), A.cols(), 0.0);
    matadd(alpha, A, 0.0, A, result);
    return result;
}

/// Return alpha*x for vectors.
[[nodiscard]] inline vec scaled(const vec &x, real alpha) {
    vec result = x;
    scale(result, alpha);
    return result;
}

/// Opt-in operator spellings for the functions above.
///
/// These are deliberately not in `num`, so that no translation unit acquires
/// them by including a header.  Write `using namespace num::ops;` to enable
/// them where expression syntax makes a formula clearer.
namespace ops {

[[nodiscard]] inline mat operator*(const mat &A, const mat &B) { return matmul(A, B); }
[[nodiscard]] inline vec operator*(const mat &A, const vec &x) { return matvec(A, x); }
[[nodiscard]] inline mat operator+(const mat &A, const mat &B) { return add(A, B); }
[[nodiscard]] inline mat operator-(const mat &A, const mat &B) { return sub(A, B); }
[[nodiscard]] inline vec operator+(const vec &x, const vec &y) { return add(x, y); }
[[nodiscard]] inline vec operator-(const vec &x, const vec &y) { return sub(x, y); }
[[nodiscard]] inline mat operator*(const mat &A, real alpha) { return scaled(A, alpha); }
[[nodiscard]] inline mat operator*(real alpha, const mat &A) { return scaled(A, alpha); }
[[nodiscard]] inline mat operator/(const mat &A, real alpha) { return scaled(A, 1.0 / alpha); }
[[nodiscard]] inline vec operator*(const vec &x, real alpha) { return scaled(x, alpha); }
[[nodiscard]] inline vec operator*(real alpha, const vec &x) { return scaled(x, alpha); }
[[nodiscard]] inline vec operator/(const vec &x, real alpha) { return scaled(x, 1.0 / alpha); }

} // namespace ops

} // namespace num
