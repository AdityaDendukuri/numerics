/// @file eigen/jacobi_eig.hpp
/// @brief Full symmetric eigendecomposition via cyclic Jacobi sweeps.
///
/// Applies orthogonal plane rotations until
/// \f$\sum_{i\ne j} A_{ij}^2 < \mathrm{tol}^2\f$.
#pragma once

#include "core/matrix.hpp"
#include "core/policy.hpp"
#include "core/vector.hpp"

namespace num {

/// @brief Symmetric eigendecomposition \f$A=V\Lambda V^T\f$.
struct EigenResult {
  Vector values;
  Matrix vectors;
  idx sweeps = 0;
  bool converged = false;
};

EigenResult eig_sym(const Matrix& A,
                    real tol = 1e-12,
                    idx max_sweeps = 100,
                    Backend backend = lapack_backend);

} // namespace num
