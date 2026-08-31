/// @file linear/eigen/jacobi_eig.hpp
/// @brief Full symmetric eigendecomposition via cyclic Jacobi sweeps.
///
/// Applies orthogonal plane rotations until
/// \f$\sum_{i\ne j} A_{ij}^2 < \mathrm{tol}^2\f$.
#pragma once

#include "linear/concepts.hpp"

#include "container/matrix.hpp"
#include "core/policy.hpp"
#include "container/vector.hpp"
#include "linear/matrix_properties.hpp"

#include <ostream>

namespace num {

/// @brief Symmetric eigendecomposition \f$A=V\Lambda V^T\f$.
struct EigenResult {
    Vector values;          ///< Eigenvalues in ascending order.
    Matrix vectors;         ///< Corresponding eigenvectors stored as columns.
    idx sweeps = 0;         ///< Jacobi sweeps for the fallback implementation.
    bool converged = false; ///< Whether the requested tolerance was met.

    friend std::ostream &operator<<(std::ostream &os, const EigenResult &r) {
        os << "EigenResult{ values: [" << r.values.size() << " eigenvalues]"
           << ", vectors: " << r.vectors.rows() << "x" << r.vectors.cols()
           << ", sweeps: " << r.sweeps
           << ", converged: " << (r.converged ? "true" : "false") << " }";
        return os;
    }
};

/// @brief Compute full symmetric eigendecomposition \f$A = V \Lambda V^T\f$.
///
/// Uses cyclic Jacobi plane rotations (in-tree fallback) or LAPACK divide-and-conquer (`dsyevd`).
/// Guarantees orthonormal eigenvectors stored as columns of \f$V\f$ and sorted real eigenvalues \f$\Lambda\f$.
///
/// @param A Symmetric matrix carrying compile-time symmetry evidence (e.g. `num::assume_symmetric(A)`).
/// @param tol Convergence tolerance on sum of squared off-diagonal elements \f$\sum_{i \ne j} A_{ij}^2\f$.
/// @param max_sweeps Maximum number of cyclic Jacobi sweeps (default: 100).
/// @param backend Execution backend (`backend::factor`, `backend::lapack`, `backend::seq`).
/// @return `EigenResult` containing sorted eigenvalues, column eigenvector matrix, sweep count, and convergence status.
/// @see assume_symmetric, make_symmetric, lanczos, power_iteration
EigenResult eig_sym(const linear::SymmetricMatrix<Matrix> &A, real tol = 1e-12, idx max_sweeps = 100,
                    Backend backend = backend::factor);

namespace unsafe {

/// @brief Symmetric eigendecomposition without requiring or checking the symmetry invariant.
///
/// Reads only the lower triangle, so an asymmetric matrix yields the spectrum of
/// its symmetric part rather than an error.
EigenResult eig_sym(const Matrix &A, real tol = 1e-12, idx max_sweeps = 100,
                    Backend backend = backend::factor);

} // namespace unsafe

/// @brief Rejects an untagged matrix at compile time.
template <class M>
requires MatrixSpace<M> && (!SymmetricMatrixLike<M>)
EigenResult eig_sym(const M & /*untagged*/, real = 1e-12, idx = 100, Backend = backend::factor) {
    static_assert(SymmetricMatrixLike<M>,
                  "eig_sym() requires a matrix carrying the symmetry invariant, which is what "
                  "guarantees a real spectrum and an orthogonal eigenbasis. "
                  "Establish it with num::assume_symmetric(A) or num::make_symmetric(A). "
                  "To bypass the invariant deliberately, call num::unsafe::eig_sym(A).");
    return {};
}

} // namespace num
