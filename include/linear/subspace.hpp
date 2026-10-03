/// @file linear/subspace.hpp
/// @brief Subspace construction and orthogonalization kernels.
#pragma once

#include "container/vector_ops.hpp"
#include <stdexcept>

#include "container/matrix.hpp"
#include "container/vector.hpp"
#include "core/types.hpp"
#include "kernel/kernel.hpp"
#include <vector>

namespace num {

/// @brief Modified Gram--Schmidt against row-major basis columns.
///
/// Keeps the sequential projection order; `project_columns` then `combine_columns` is the
/// faster classical variant, with weaker stability.
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

} // namespace num

namespace num::dispatch::subspace {

/// @brief Modified Gram–Schmidt orthogonalization against basis vectors \f$\mathbf{v}_0, \dots,
/// \mathbf{v}_{k-1}\f$.
[[nodiscard]] real mgs_orthogonalize(const array<vec<real>> &basis, vec<real> &v,
                                     array<real> &h, idx k);

/// @brief Modified Gram–Schmidt orthogonalization against columns \f$0, \dots, k-1\f$ of a
/// row-major matrix.
[[nodiscard]] real mgs_orthogonalize(const mat<real> &basis, idx k, vec<real> &v);

/// @brief One Arnoldi iteration step: expands orthonormal Krylov basis \f$V_k \to V_{k+1}\f$.
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
