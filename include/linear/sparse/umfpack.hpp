/// @file umfpack.hpp
/// @brief Optional SuiteSparse UMFPACK factorizations (real, and complex shifted), with transposed solves.
///
/// UMFPACK is multifrontal with dense BLAS kernels, so it factors high-fill sparse matrices several
/// times faster than KLU. It is GPL-2.0-or-later: the backend is off unless numerics is configured with
/// `-DNUMERICS_USE_UMFPACK=ON`, and binaries built with it are subject to the GPL when distributed.
#pragma once

#include "container/vector.hpp"
#include "core/types.hpp"
#include "linear/sparse/sparse.hpp"
#include <memory>

namespace num {

/// True when numerics was built with the optional UMFPACK backend.
[[nodiscard]] bool umfpack_available() noexcept;

/// @brief Sparse LU of a square real matrix by UMFPACK.
///
/// No iterative refinement; the solve workspace is preallocated, so one factor serves one thread at a time.
class umfpack_factorization {
  public:
    /// Factor a square CSR matrix; throws when UMFPACK is unavailable or the matrix is singular.
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

/// @brief Solve \f$Ax = b\f$.
void solve(const umfpack_factorization &factor, const vec<real> &b, vec<real> &x);
/// @brief Solve \f$A^T x = b\f$.
void solve_transpose(const umfpack_factorization &factor, const vec<real> &b, vec<real> &x);

/// @brief Sparse LU of \f$sI - A\f$ for a real A and complex shift s, by UMFPACK's complex interface.
///
/// A must store its diagonal explicitly. The transposed solve is the non-conjugate transpose, so one
/// factorization serves both a forward resolvent and its adjoint.
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

/// @brief Solve \f$(sI - A)x = b\f$.
void solve(const umfpack_shifted_lu &factor, const vec<cplx> &b, vec<cplx> &x);
/// @brief Solve \f$(sI - A)^T x = b\f$ (no conjugation).
void solve_transpose(const umfpack_shifted_lu &factor, const vec<cplx> &b, vec<cplx> &x);

} // namespace num
