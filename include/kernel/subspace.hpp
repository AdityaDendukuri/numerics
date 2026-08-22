/// @file kernel/subspace.hpp
/// @brief Subspace construction and orthogonalization kernels.
#pragma once

#include "core/matrix.hpp"
#include "core/types.hpp"
#include "core/vector.hpp"
#include <vector>

namespace num::kernel::subspace {

/// @brief Modified Gram-Schmidt against basis[0..k-1].
[[nodiscard]] real mgs_orthogonalize(const std::vector<Vector> &basis, Vector &v,
                                     std::vector<real> &h, idx k);

/// @brief Modified Gram-Schmidt against columns 0..k-1 of a row-major matrix.
[[nodiscard]] real mgs_orthogonalize(const Matrix &basis, idx k, Vector &v);

/// @brief One Arnoldi step: expand the orthonormal basis by one vector.
template <class Op>
requires requires(const Op &A, const Vector &x, Vector &y) {
    A.apply(x, y);
}
[[nodiscard]] real arnoldi_step(const Op &A, std::vector<Vector> &basis, std::vector<real> &h,
                                idx k, Vector &scratch, real breakdown_tol = real(1e-14)) {
    A.apply(basis[k], scratch);

    const real beta = mgs_orthogonalize(basis, scratch, h, k + 1);
    h[k + 1] = beta;

    if (beta > breakdown_tol) {
        scale(scratch, real(1) / beta);
        basis.push_back(scratch);
    }

    return beta;
}

} // namespace num::kernel::subspace
