/// @file umfpack.hpp
/// @brief Optional SuiteSparse UMFPACK factorization for real sparse matrices.
#pragma once

#include "container/matrix.hpp"
#include "container/vector.hpp"
#include "linear/solve.hpp"
#include "linear/sparse/sparse.hpp"
#include <memory>

namespace num {

/// True when Numerics was built with the optional SuiteSparse UMFPACK backend.
[[nodiscard]] bool umfpack_available() noexcept;

/// Reusable sparse LU factorization backed by SuiteSparse UMFPACK.
class umfpack_factor {
  public:
    /// Factor a square CSR matrix; throws when UMFPACK is unavailable or factorization
    /// fails.
    explicit umfpack_factor(const spmat &matrix);
    ~umfpack_factor();
    umfpack_factor(umfpack_factor &&) noexcept;
    umfpack_factor &operator=(umfpack_factor &&) noexcept;
    umfpack_factor(const umfpack_factor &) = delete;
    umfpack_factor &operator=(const umfpack_factor &) = delete;

    /// Return the order of the factored matrix.
    [[nodiscard]] idx size() const noexcept;

  private:
    friend void solve(const umfpack_factor &, const vec<real> &, vec<real> &);
    friend void solve(const umfpack_factor &, const mat<real> &, mat<real> &);
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

/// @brief Solve \f$Ax = b\f$. `x` may be `b`.
void solve(const umfpack_factor &factor, const vec<real> &b, vec<real> &x);
/// @brief Solve \f$AX = B\f$. `X` may be `B`.
void solve(const umfpack_factor &factor, const mat<real> &B, mat<real> &X);

} // namespace num
