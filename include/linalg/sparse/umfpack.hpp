/// @file umfpack.hpp
/// @brief Optional SuiteSparse UMFPACK factorization for real sparse matrices.
#pragma once

#include "core/matrix.hpp"
#include "core/vector.hpp"
#include "linalg/sparse/sparse.hpp"
#include <memory>

namespace num {

[[nodiscard]] bool umfpack_available() noexcept;

class UMFPACKFactor {
public:
  explicit UMFPACKFactor(const SparseMatrix& matrix);
  ~UMFPACKFactor();
  UMFPACKFactor(UMFPACKFactor&&) noexcept;
  UMFPACKFactor& operator=(UMFPACKFactor&&) noexcept;
  UMFPACKFactor(const UMFPACKFactor&) = delete;
  UMFPACKFactor& operator=(const UMFPACKFactor&) = delete;

  [[nodiscard]] idx size() const noexcept;
  void solve(const Vector& rhs, Vector& solution) const;
  void solve(const Matrix& rhs, Matrix& solution) const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace num
