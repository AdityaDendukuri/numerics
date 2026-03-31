/// @file eigen/jacobi_eig.hpp
/// @brief Full symmetric eigendecomposition via cyclic Jacobi sweeps
///
/// Computes ALL eigenvalues and eigenvectors of a real symmetric matrix.
///
/// Algorithm  -- cyclic Jacobi:
///   Repeatedly apply plane (Givens) rotations in the (p,q) plane to zero
///   the off-diagonal element A[p,q].  Each rotation is a similarity transform
///   so eigenvalues are preserved.  Convergence is quadratic: after each full
///   sweep the sum of squared off-diagonal elements decreases by at least a
///   constant factor.
///
/// Why Jacobi instead of the implicit QR algorithm?
///   Jacobi is conceptually simpler and each rotation is the exact same
///   2x2 eigendecomposition students encounter first.  For dense n < 1000 the
///   performance is acceptable.  LAPACK uses the Francis QR algorithm (dsyev)
///   which is O(n^2) per iteration vs O(n^2) per sweep here, but harder to
///   explain.
///
/// Complexity: O(n^2) per sweep, O(n) sweeps typical -> O(n^3) total.
#pragma once

#include "core/matrix.hpp"
#include "core/policy.hpp"
#include "core/vector.hpp"

namespace num {

/// @brief Full eigendecomposition result: A = V * diag(values) * V^T
struct EigenResult {
    Vector values;  ///< Eigenvalues in ascending order
    Matrix vectors; ///< Eigenvectors as columns of an nxn orthogonal matrix
    idx    sweeps    = 0;     ///< Number of Jacobi sweeps performed
    bool   converged = false; ///< Whether off-diagonal norm fell below tol
};

/// @brief Full eigendecomposition of a real symmetric matrix.
///
/// The rotation accumulation loop is parallelised when backend == Backend::omp.
///
/// @param A          Real symmetric nxn matrix (upper/lower triangle used)
/// @param tol        Jacobi convergence tolerance (ignored for Backend::lapack)
/// @param max_sweeps Jacobi sweep cap (ignored for Backend::lapack)
/// @param backend    Backend::lapack uses LAPACKE_dsyevd (default when
/// available).
///                   Backend::omp    parallelises the Jacobi rotation loop.
///                   Backend::seq    uses our cyclic Jacobi implementation.
/// @return EigenResult with eigenvalues in ascending order and corresponding
///         eigenvectors as columns of the V matrix (A = V diag(lambda) V^T)
EigenResult eig_sym(const Matrix& A,
                    real          tol        = 1e-12,
                    idx           max_sweeps = 100,
                    Backend       backend    = lapack_backend);

} // namespace num
