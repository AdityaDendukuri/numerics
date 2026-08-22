/// @file svd/svd.hpp
/// @brief Dense and randomized truncated SVD.
#pragma once

#include "core/matrix.hpp"
#include "core/policy.hpp"
#include "core/util/math.hpp"
#include "core/vector.hpp"
#include "linalg/factorization/qr.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace num {

/// Singular value decomposition and convergence metadata.
struct SVDResult {
    Matrix U;               ///< Left singular vectors.
    Vector S;               ///< Singular values in descending order.
    Matrix Vt;              ///< Transposed right singular vectors.
    idx sweeps = 0;         ///< Jacobi sweeps for the fallback implementation.
    bool converged = false; ///< Whether the requested tolerance was met.
};

/// Compute a full dense singular value decomposition A=U diag(S) V^T.
SVDResult svd(const Matrix &A, Backend backend = lapack_backend, real tol = 1e-12,
              idx max_sweeps = 100);

/// Compute a randomized rank-k approximation with optional reproducible RNG.
SVDResult svd_truncated(const Matrix &A, idx k, Backend backend = default_backend,
                        idx oversampling = 10, Rng *rng = nullptr);

} // namespace num
