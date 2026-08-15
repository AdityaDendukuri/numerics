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

struct SVDResult {
  Matrix U;
  Vector S;
  Matrix Vt;
  idx sweeps = 0;
  bool converged = false;
};

SVDResult svd(const Matrix& A,
              Backend backend = lapack_backend,
              real tol = 1e-12,
              idx max_sweeps = 100);

SVDResult svd_truncated(const Matrix& A,
                        idx k,
                        Backend backend = default_backend,
                        idx oversampling = 10,
                        Rng* rng = nullptr);

} // namespace num
