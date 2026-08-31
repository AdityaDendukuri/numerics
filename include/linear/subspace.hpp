/// @file linear/subspace.hpp
/// @brief Subspace construction and orthogonalization kernels.
#pragma once

#include "container/vector_ops.hpp"
#include <stdexcept>

#include "container/matrix.hpp"
#include "container/vector.hpp"
#include "core/types.hpp"
#include "kernel/raw.hpp"
#include <vector>

namespace num::kernel::subspace {

/// @brief Modified Gram–Schmidt orthogonalization against basis vectors \f$\mathbf{v}_0, \dots,
/// \mathbf{v}_{k-1}\f$.
[[nodiscard]] real mgs_orthogonalize(const std::vector<Vector> &basis, Vector &v,
                                     std::vector<real> &h, idx k);

/// @brief Modified Gram–Schmidt orthogonalization against columns \f$0, \dots, k-1\f$ of a
/// row-major matrix.
[[nodiscard]] real mgs_orthogonalize(const Matrix &basis, idx k, Vector &v);

/// @brief One Arnoldi iteration step: expands orthonormal Krylov basis \f$V_k \to V_{k+1}\f$.
template <class Op>
requires requires(const Op &A, const Vector &x, Vector &y) {
    A.apply(x, y);
}
[[nodiscard]] real arnoldi_step(const Op &A, std::vector<Vector> &basis, std::vector<real> &h,
                                idx k, Vector &scratch, real breakdown_tol = real(1e-14)) {
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

inline real mgs_orthogonalize(const std::vector<Vector> &basis, Vector &v, std::vector<real> &h,
                              idx k) {
    for (idx i = 0; i < k; ++i) {
        // h_i <- v_i^T*v
        h[i] = dot(v, basis[i]);
        // v <- v - h_i*v_i
        axpy(-h[i], basis[i], v);
    }
    return norm(v);
}

inline real mgs_orthogonalize(const Matrix &basis, idx k, Vector &v) {
    const idx n = basis.rows();
    // v <- (I - V_k*V_k^T)v, in modified Gram--Schmidt order
    raw::mgs_columns(v.data(), basis.data(), basis.cols(), n, k);
    return norm(v);
}

} // namespace num::kernel::subspace
