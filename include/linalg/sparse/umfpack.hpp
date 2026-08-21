/// @file umfpack.hpp
/// @brief Optional SuiteSparse UMFPACK factorization for real sparse matrices.
#pragma once

#include "core/matrix.hpp"
#include "core/vector.hpp"
#include "linalg/sparse/sparse.hpp"
#include <memory>

namespace num {

/// True when Numerics was built with the optional SuiteSparse UMFPACK backend.
[[nodiscard]] bool umfpack_available() noexcept;

/// Reusable sparse LU factorization backed by SuiteSparse UMFPACK.
class UMFPACKFactor {
public:
  /// Factor a square CSR matrix; throws when UMFPACK is unavailable or factorization
  /// fails.
  explicit UMFPACKFactor(const SparseMatrix& matrix);
  ~UMFPACKFactor();
  UMFPACKFactor(UMFPACKFactor&&) noexcept;
  UMFPACKFactor& operator=(UMFPACKFactor&&) noexcept;
  UMFPACKFactor(const UMFPACKFactor&) = delete;
  UMFPACKFactor& operator=(const UMFPACKFactor&) = delete;

  /// Return the order of the factored matrix.
  [[nodiscard]] idx size() const noexcept;
  /// Solve Ax=B for one or more dense right-hand sides.
  void solve(const Vector& rhs, Vector& solution) const;
  void solve(const Matrix& rhs, Matrix& solution) const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace num
