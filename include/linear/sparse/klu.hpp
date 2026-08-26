/// @file klu.hpp
/// @brief Optional SuiteSparse KLU factorization for real sparse matrices.
#pragma once

#include "container/matrix.hpp"
#include "container/vector.hpp"
#include "linear/sparse/sparse.hpp"
#include <memory>

namespace num {

/// True when Numerics was built with the optional SuiteSparse KLU backend.
[[nodiscard]] bool klu_available() noexcept;

/// Reusable sparse LU factorization backed by SuiteSparse KLU.
class KLUFactor {
  public:
    /// Factor a square CSR matrix; throws when KLU is unavailable or factorization fails.
    explicit KLUFactor(const SparseMatrix &matrix);
    ~KLUFactor();
    KLUFactor(KLUFactor &&) noexcept;
    KLUFactor &operator=(KLUFactor &&) noexcept;
    KLUFactor(const KLUFactor &) = delete;
    KLUFactor &operator=(const KLUFactor &) = delete;

    /// Return the order of the factored matrix.
    [[nodiscard]] idx size() const noexcept;
    /// Solve Ax=B for one or more dense right-hand sides.
    void solve(const Vector &rhs, Vector &solution) const;
    void solve(const Matrix &rhs, Matrix &solution) const;
    /// Solve A^T x=b.
    void solve_transpose(const Vector &rhs, Vector &solution) const;
    /// Solve A^T X=B for several dense right-hand sides.
    void solve_transpose(const Matrix &rhs, Matrix &solution) const;
    /// Replace one or more right-hand sides with their solutions.
    void solve_in_place(Vector &right_hand_side) const;
    void solve_in_place(Matrix &right_hand_sides) const;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace num
