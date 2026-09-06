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
class klu_factorization {
  public:
    /// Factor a square CSR matrix; throws when KLU is unavailable or factorization fails.
    explicit klu_factorization(const spmat &matrix);
    ~klu_factorization();
    klu_factorization(klu_factorization &&) noexcept;
    klu_factorization &operator=(klu_factorization &&) noexcept;
    klu_factorization(const klu_factorization &) = delete;
    klu_factorization &operator=(const klu_factorization &) = delete;

    /// Return the order of the factored matrix.
    [[nodiscard]] idx size() const noexcept;
    /// Solve Ax=B for one or more dense right-hand sides.
    void solve(const vec &rhs, vec &solution) const;
    void solve(const mat &rhs, mat &solution) const;
    /// Solve A^T x=b.
    void solve_transpose(const vec &rhs, vec &solution) const;
    /// Solve A^T X=B for several dense right-hand sides.
    void solve_transpose(const mat &rhs, mat &solution) const;
    /// Replace one or more right-hand sides with their solutions.
    void solve_in_place(vec &right_hand_side) const;
    void solve_in_place(mat &right_hand_sides) const;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace num
