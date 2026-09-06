/// @file umfpack.hpp
/// @brief Optional SuiteSparse UMFPACK factorization for real sparse matrices.
#pragma once

#include "container/matrix.hpp"
#include "container/vector.hpp"
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
    /// Solve Ax=B for one or more dense right-hand sides.
    void solve(const vec &rhs, vec &solution) const;
    void solve(const mat &rhs, mat &solution) const;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace num
