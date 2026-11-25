/// @file linear/sparse/klu.hpp
/// @brief Sparse LU through SuiteSparse KLU.
#pragma once

#include "container/matrix.hpp"
#include "container/vector.hpp"
#include "linear/solve.hpp"
#include "linear/sparse/sparse.hpp"
#include <memory>

namespace num {

/// @brief Whether KLU is available.
[[nodiscard]] bool klu_available() noexcept;

/// @brief KLU sparse LU: \f$P A Q = L U\f$.
class klu_factorization {
  public:
    /// @brief Factor a square sparse matrix.
    /// @throws std::runtime_error If KLU is unavailable or the factorization fails.
    explicit klu_factorization(const spmat &matrix);
    ~klu_factorization();
    klu_factorization(klu_factorization &&) noexcept;
    klu_factorization &operator=(klu_factorization &&) noexcept;
    klu_factorization(const klu_factorization &) = delete;
    klu_factorization &operator=(const klu_factorization &) = delete;

    /// @brief Order \f$n\f$.
    [[nodiscard]] idx size() const noexcept;

  private:
    friend void solve(const klu_factorization &, const vec<real> &, vec<real> &);
    friend void solve(const klu_factorization &, const mat<real> &, mat<real> &);
    friend void solve_transpose(const klu_factorization &, const vec<real> &, vec<real> &);
    friend void solve_transpose(const klu_factorization &, const mat<real> &, mat<real> &);
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

/// @brief KLU solve: \f$x \leftarrow A^{-1} b\f$. `x` may alias `b`.
void solve(const klu_factorization &factor, const vec<real> &b, vec<real> &x);
/// @brief KLU solve: \f$X \leftarrow A^{-1} B\f$. `X` may alias `B`.
void solve(const klu_factorization &factor, const mat<real> &B, mat<real> &X);
/// @brief KLU transposed solve: \f$x \leftarrow A^{-T} b\f$. `x` may alias `b`.
void solve_transpose(const klu_factorization &factor, const vec<real> &b, vec<real> &x);
void solve_transpose(const klu_factorization &factor, const mat<real> &B, mat<real> &X);

} // namespace num
