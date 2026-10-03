/// @file klu.hpp
/// @brief Optional SuiteSparse KLU factorization for real sparse matrices.
#pragma once

#include "container/matrix.hpp"
#include "container/vector.hpp"
#include "linear/solve.hpp"
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

  private:
    friend void solve(const klu_factorization &, const vec<real> &, vec<real> &);
    friend void solve(const klu_factorization &, const mat<real> &, mat<real> &);
    friend void solve_transpose(const klu_factorization &, const vec<real> &, vec<real> &);
    friend void solve_transpose(const klu_factorization &, const mat<real> &, mat<real> &);
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

/// @brief Solve \f$Ax = b\f$. `x` may be `b`.
void solve(const klu_factorization &factor, const vec<real> &b, vec<real> &x);
/// @brief Solve \f$AX = B\f$. `X` may be `B`.
void solve(const klu_factorization &factor, const mat<real> &B, mat<real> &X);
/// @brief Solve \f$A^Tx = b\f$. `x` may be `b`.
void solve_transpose(const klu_factorization &factor, const vec<real> &b, vec<real> &x);
void solve_transpose(const klu_factorization &factor, const mat<real> &B, mat<real> &X);

} // namespace num
