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
/// const Vector r = y - Z * q;
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
[[nodiscard]] inline Matrix matmul(const Matrix &A, const Matrix &B) {
    if (A.cols() != B.rows()) {
        throw std::invalid_argument("matmul: inner dimensions do not agree");
    }
    Matrix C(A.rows(), B.cols(), 0.0);
    matmul(A, B, C);
    return C;
}

/// Return A*x. Allocates; prefer the out-param form in hot loops.
[[nodiscard]] inline Vector matvec(const Matrix &A, const Vector &x) {
    if (A.cols() != x.size()) {
        throw std::invalid_argument("matvec: matrix columns do not match vector size");
    }
    Vector y(A.rows(), 0.0);
    matvec(A, x, y);
    return y;
}

/// Return A+B. Allocates; prefer the out-param form in hot loops.
[[nodiscard]] inline Matrix add(const Matrix &A, const Matrix &B) {
    if (A.rows() != B.rows() || A.cols() != B.cols()) {
        throw std::invalid_argument("add: matrix shapes do not agree");
    }
    Matrix C(A.rows(), A.cols(), 0.0);
    matadd(1.0, A, 1.0, B, C);
    return C;
}

/// Return A-B. Allocates; prefer the out-param form in hot loops.
[[nodiscard]] inline Matrix sub(const Matrix &A, const Matrix &B) {
    if (A.rows() != B.rows() || A.cols() != B.cols()) {
        throw std::invalid_argument("sub: matrix shapes do not agree");
    }
    Matrix C(A.rows(), A.cols(), 0.0);
    matadd(1.0, A, -1.0, B, C);
    return C;
}

/// Return x+y. Allocates; prefer the out-param form in hot loops.
[[nodiscard]] inline Vector add(const Vector &x, const Vector &y) {
    if (x.size() != y.size()) {
        throw std::invalid_argument("add: vector sizes do not agree");
    }
    Vector z(x.size(), 0.0);
    add(x, y, z);
    return z;
}

/// Return x-y. Allocates; prefer the out-param form in hot loops.
[[nodiscard]] inline Vector sub(const Vector &x, const Vector &y) {
    if (x.size() != y.size()) {
        throw std::invalid_argument("sub: vector sizes do not agree");
    }
    Vector z(x);
    axpy(-1.0, y, z);
    return z;
}

/// Return alpha*A, matching the value-returning `scaled` for sparse matrices.
/// Allocates; prefer in-place scaling in hot loops.
[[nodiscard]] inline Matrix scaled(const Matrix &A, real alpha) {
    Matrix result(A.rows(), A.cols(), 0.0);
    matadd(alpha, A, 0.0, A, result);
    return result;
}

/// Return alpha*x for vectors.
[[nodiscard]] inline Vector scaled(const Vector &x, real alpha) {
    Vector result = x;
    scale(result, alpha);
    return result;
}

/// Opt-in operator spellings for the functions above.
///
/// These are deliberately not in `num`, so that no translation unit acquires
/// them by including a header.  Write `using namespace num::ops;` to enable
/// them where expression syntax makes a formula clearer.
namespace ops {

[[nodiscard]] inline Matrix operator*(const Matrix &A, const Matrix &B) { return matmul(A, B); }
[[nodiscard]] inline Vector operator*(const Matrix &A, const Vector &x) { return matvec(A, x); }
[[nodiscard]] inline Matrix operator+(const Matrix &A, const Matrix &B) { return add(A, B); }
[[nodiscard]] inline Matrix operator-(const Matrix &A, const Matrix &B) { return sub(A, B); }
[[nodiscard]] inline Vector operator+(const Vector &x, const Vector &y) { return add(x, y); }
[[nodiscard]] inline Vector operator-(const Vector &x, const Vector &y) { return sub(x, y); }
[[nodiscard]] inline Matrix operator*(const Matrix &A, real alpha) { return scaled(A, alpha); }
[[nodiscard]] inline Matrix operator*(real alpha, const Matrix &A) { return scaled(A, alpha); }
[[nodiscard]] inline Matrix operator/(const Matrix &A, real alpha) { return scaled(A, 1.0 / alpha); }
[[nodiscard]] inline Vector operator*(const Vector &x, real alpha) { return scaled(x, alpha); }
[[nodiscard]] inline Vector operator*(real alpha, const Vector &x) { return scaled(x, alpha); }
[[nodiscard]] inline Vector operator/(const Vector &x, real alpha) { return scaled(x, 1.0 / alpha); }

} // namespace ops

} // namespace num
