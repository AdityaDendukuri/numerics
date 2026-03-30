/// @file svd/svd.hpp
/// @brief Singular Value Decomposition  -- dense and randomized truncated
///
/// Two algorithms:
///
///   svd(A)             Full SVD via one-sided Jacobi: A = U * diag(S) * V^T
///   svd_truncated(A,k) Randomized truncated SVD: top-k singular triplets
///
/// Full SVD (one-sided Jacobi)
///
/// The one-sided Jacobi algorithm applies Givens rotations to the columns
/// of A (from the right) until the columns are mutually orthogonal.
///
/// Randomized truncated SVD
///
/// For large A where only the top-k singular triplets are needed.
/// Algorithm (Halko, Martinsson, Tropp 2011):
///
///   1. Draw Omega in R^{nx(k+p)}  (Gaussian random sketch, p=10 oversampling)
///   2. Y = A * Omega              (sample the range of A)
///   3. Q = QR(Y)              (orthonormal basis for approx. range)
///   4. B = Q^T * A             (project A to k-dim subspace)
///   5. SVD of B = U~ * Sigma * V^T  (small kxn problem)
///   6. U = Q * U~              (lift back to full dimension)
///
/// Cost: O(mnk) vs O(mn*min(m,n)) for full SVD.
#pragma once

#include "core/matrix.hpp"
#include "core/vector.hpp"
#include "core/policy.hpp"
#include "linalg/factorization/qr.hpp"
#include "core/util/math.hpp"
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace num {

/// @brief Result of a Singular Value Decomposition: A = U * diag(S) * V^T
struct SVDResult {
    Matrix U;         ///< mxr left singular vectors (columns orthonormal)
    Vector S;         ///< r singular values in descending order
    Matrix Vt;        ///< rxn right singular vectors (rows orthonormal)
    idx    sweeps;    ///< Jacobi sweeps (full SVD only; 0 for randomized)
    bool   converged; ///< Whether Jacobi converged (always true for randomized)
};

/// @brief Full SVD of an mxn matrix.
///
/// Returns the economy SVD: r = min(m,n), U is mxr, S has r elements, Vt is rxn.
///
/// @param A          Input matrix (not modified)
/// @param backend    Backend::lapack uses LAPACKE_dgesdd (default when available).
///                   Backend::omp    parallelises Jacobi column-update loops.
///                   Backend::seq    uses our one-sided Jacobi implementation.
/// @param tol        Jacobi convergence tolerance (ignored for Backend::lapack)
/// @param max_sweeps Jacobi sweep cap (ignored for Backend::lapack)
SVDResult svd(const Matrix& A,
              Backend backend = lapack_backend,
              real tol = 1e-12, idx max_sweeps = 100);

/// @brief Randomized truncated SVD  -- top-k singular triplets.
///
/// Efficient when k << min(m,n).  The two dominant costs  -- Y = A*Omega and B = Q^T*A
///  -- are dispatched via the given backend.
///
/// @param A            Input matrix
/// @param k            Number of singular values/vectors to compute
/// @param backend      Backend for internal matmul calls (default: default_backend)
/// @param oversampling Extra random vectors for accuracy (default 10)
/// @param rng          Random number generator (default: seeded from hardware)
SVDResult svd_truncated(const Matrix& A, idx k,
                        Backend backend = default_backend,
                        idx oversampling = 10, Rng* rng = nullptr);

} // namespace num
