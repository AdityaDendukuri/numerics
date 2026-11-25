/// @file linear/subspace.hpp
/// @brief Orthogonalization and Krylov basis construction.
#pragma once

#include "container/matrix_ops.hpp"
#include "container/vector_ops.hpp"
#include <stdexcept>

#include "container/matrix.hpp"
#include "container/vector.hpp"
#include "core/types.hpp"
#include "kernel/kernel.hpp"
#include <vector>

namespace num {

/// @brief Modified Gram–Schmidt: \f$v \leftarrow v - (q_j^T v) q_j\f$ for each column in turn.
template <std::floating_point T>
inline void mgs_columns(T *NUM_K_RESTRICT v, const T *NUM_K_RESTRICT basis, idx ldb, idx rows,
                        idx columns, T *coefficients = nullptr) noexcept {
    for (idx column = 0; column < columns; ++column) {
        T projection = T(0);
        for (idx row = 0; row < rows; ++row) {
            projection += basis[(row * ldb) + column] * v[row];
        }
        if (coefficients != nullptr) {
            coefficients[column] = projection;
        }
        for (idx row = 0; row < rows; ++row) {
            v[row] -= projection * basis[(row * ldb) + column];
        }
    }
}

/// @brief Classical Gram–Schmidt, twice (CGS2): \f$v \leftarrow (I - Q_k^T Q_k)^2 v\f$.
/// Returns the coefficients; rows of \f$Q_k\f$ must be orthonormal.
inline array<real> cgs2_orthogonalize(const mat<real> &Q, idx k, vec<real> &v) {
    array<real> h(k, 0.0);
    vec<real> t(k, 0.0);
    for (int pass = 0; pass < 2; ++pass) {
        gemv(1.0, Q, false, v, 0.0, t, k);
        gemv(-1.0, Q, true, t, 1.0, v, k);
        for (idx i = 0; i < k; ++i) {
            h[i] += t[i];
        }
    }
    return h;
}

} // namespace num

namespace num::dispatch::subspace {

/// @brief Modified Gram–Schmidt against \f$v_0, \ldots, v_{k-1}\f$.
[[nodiscard]] real mgs_orthogonalize(const array<vec<real>> &basis, vec<real> &v,
                                     array<real> &h, idx k);

/// @brief Modified Gram–Schmidt against columns \f$0, \ldots, k-1\f$ of a row-major matrix.
[[nodiscard]] real mgs_orthogonalize(const mat<real> &basis, idx k, vec<real> &v);

/// @brief Arnoldi step: \f$A v_k = \sum_{j \le k+1} h_{jk} v_j\f$, extending the basis by one.
template <class Op>
requires requires(const Op &A, const vec<real> &x, vec<real> &y) {
    A.apply(x, y);
}
[[nodiscard]] real arnoldi_step(const Op &A, array<vec<real>> &basis, array<real> &h,
                                idx k, vec<real> &scratch, real breakdown_tol = real(1e-14)) {
    // w <- A*v_k
    A.apply(basis[k], scratch);

    // (h_{0:k,k}, h_{k+1,k}) <- Arnoldi orthogonalization of w
    const real beta = mgs_orthogonalize(basis, scratch, h, k + 1);
    h[k + 1] = beta;

    if (beta > breakdown_tol) {
        // v_{k+1} <- w/h_{k+1,k}
        scale(scratch, real(1) / beta);
        basis.push_back(scratch);
    }

    return beta;
}

inline real mgs_orthogonalize(const array<vec<real>> &basis, vec<real> &v, array<real> &h,
                              idx k) {
    for (idx i = 0; i < k; ++i) {
        // h_i <- v_i^T*v
        h[i] = dot(v, basis[i]);
        // v <- v - h_i*v_i
        axpy(-h[i], basis[i], v);
    }
    return norm(v);
}

inline real mgs_orthogonalize(const mat<real> &basis, idx k, vec<real> &v) {
    const idx n = basis.rows();
    // v <- (I - V_k*V_k^T)v, in modified Gram--Schmidt order
    num::mgs_columns(v.data(), basis.data(), basis.cols(), n, k);
    return norm(v);
}

} // namespace num::dispatch::subspace
