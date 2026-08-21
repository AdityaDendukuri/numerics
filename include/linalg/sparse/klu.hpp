/// @file klu.hpp
/// @brief Optional SuiteSparse KLU factorization for real sparse matrices.
#pragma once

#include "core/matrix.hpp"
#include "core/vector.hpp"
#include "linalg/sparse/sparse.hpp"
#include <memory>

namespace num {

/// True when Numerics was built with the optional SuiteSparse KLU backend.
[[nodiscard]] bool klu_available() noexcept;

/// Reusable sparse LU factorization backed by SuiteSparse KLU.
class KLUFactor {
public:
  explicit KLUFactor(const SparseMatrix& matrix);
  ~KLUFactor();
  KLUFactor(KLUFactor&&) noexcept;
  KLUFactor& operator=(KLUFactor&&) noexcept;
  KLUFactor(const KLUFactor&) = delete;
  KLUFactor& operator=(const KLUFactor&) = delete;

  [[nodiscard]] idx size() const noexcept;
  void solve(const Vector& rhs, Vector& solution) const;
  void solve(const Matrix& rhs, Matrix& solution) const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace num
