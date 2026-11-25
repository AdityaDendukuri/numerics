/// @file linear/sparse/umfpack.hpp
/// @brief Sparse LU through SuiteSparse UMFPACK, real and complex-shifted.
/// @warning UMFPACK is GPL-2.0-or-later: binaries built with `-DNUMERICS_USE_UMFPACK=ON`
/// are subject to the GPL.
#pragma once

#include "container/vector.hpp"
#include "core/types.hpp"
#include "linear/sparse/sparse.hpp"
#include <memory>

namespace num {

/// @brief Whether UMFPACK is available.
[[nodiscard]] bool umfpack_available() noexcept;

/// @brief UMFPACK sparse LU: \f$P A Q = L U\f$. One factor serves one thread at a time.
class umfpack_factorization {
  public:
    /// @brief Factor a square sparse matrix.
    /// @throws std::runtime_error If UMFPACK is unavailable or `A` is singular.
    explicit umfpack_factorization(const spmat &matrix);
    ~umfpack_factorization();
    umfpack_factorization(umfpack_factorization &&) noexcept;
    umfpack_factorization &operator=(umfpack_factorization &&) noexcept;
    umfpack_factorization(const umfpack_factorization &) = delete;
    umfpack_factorization &operator=(const umfpack_factorization &) = delete;

    [[nodiscard]] idx size() const noexcept;

  private:
    friend void solve(const umfpack_factorization &, const vec<real> &, vec<real> &);
    friend void solve_transpose(const umfpack_factorization &, const vec<real> &, vec<real> &);
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

/// @brief UMFPACK solve: \f$x \leftarrow A^{-1} b\f$.
void solve(const umfpack_factorization &factor, const vec<real> &b, vec<real> &x);
/// @brief UMFPACK transposed solve: \f$x \leftarrow A^{-T} b\f$.
void solve_transpose(const umfpack_factorization &factor, const vec<real> &b, vec<real> &x);

/// @brief UMFPACK shifted LU: \f$s I - A = L U\f$, real \f$A\f$, complex \f$s\f$.
/// The diagonal of `A` must be stored.
class umfpack_shifted_lu {
  public:
    umfpack_shifted_lu(const spmat &A, cplx s);
    ~umfpack_shifted_lu();
    umfpack_shifted_lu(umfpack_shifted_lu &&) noexcept;
    umfpack_shifted_lu &operator=(umfpack_shifted_lu &&) noexcept;
    umfpack_shifted_lu(const umfpack_shifted_lu &) = delete;
    umfpack_shifted_lu &operator=(const umfpack_shifted_lu &) = delete;

    [[nodiscard]] idx size() const noexcept;

  private:
    friend void solve(const umfpack_shifted_lu &, const vec<cplx> &, vec<cplx> &);
    friend void solve_transpose(const umfpack_shifted_lu &, const vec<cplx> &, vec<cplx> &);
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

/// @brief Shifted solve: \f$x \leftarrow (s I - A)^{-1} b\f$.
void solve(const umfpack_shifted_lu &factor, const vec<cplx> &b, vec<cplx> &x);
/// @brief Shifted transposed solve, unconjugated: \f$x \leftarrow (s I - A)^{-T} b\f$.
void solve_transpose(const umfpack_shifted_lu &factor, const vec<cplx> &b, vec<cplx> &x);

} // namespace num
