/// @file linear/solvers/sparse_resolvent.hpp
/// @brief Shifted sparse solves through a direct backend.
#pragma once

#include "container/vector.hpp"
#include "core/types.hpp"
#include "linear/solve.hpp"
#include "linear/sparse/sparse.hpp"
#include <memory>
#include <vector>

namespace num {

/// @brief Whether KLU is available for shifted factorizations.
[[nodiscard]] bool sparse_resolvent_available() noexcept;

class sparse_shifted_lu;

/// @brief Sparse resolvent: symbolic analysis of \f$s I - A\f$ once, numeric factor per shift.
/// Requires KLU; `shift` is thread-safe on one analysis.
/// @note Without KLU, `shift` throws rather than densifying a large matrix.
class sparse_resolvent {
  public:
    /// @brief Analyze the pattern of `A`; its diagonal must be stored.
    explicit sparse_resolvent(const spmat &A);

    [[nodiscard]] idx size() const noexcept;

    struct analysis;

  private:
    std::shared_ptr<const analysis> analysis_;
    friend class sparse_shifted_lu;
    friend sparse_shifted_lu shift(const sparse_resolvent &R, cplx s);
};

/// @brief Shifted sparse factor: \f$s I - A\f$ for one \f$s\f$.
class sparse_shifted_lu {
  public:
    ~sparse_shifted_lu();
    sparse_shifted_lu(sparse_shifted_lu &&) noexcept;
    sparse_shifted_lu &operator=(sparse_shifted_lu &&) noexcept;
    sparse_shifted_lu(const sparse_shifted_lu &) = delete;
    sparse_shifted_lu &operator=(const sparse_shifted_lu &) = delete;

    [[nodiscard]] idx size() const noexcept;

  private:
    struct numeric;
    explicit sparse_shifted_lu(std::unique_ptr<numeric> impl);
    std::unique_ptr<numeric> impl_;
    friend sparse_shifted_lu shift(const sparse_resolvent &R, cplx s);
    friend void solve(const sparse_shifted_lu &F, const vec<cplx> &b, vec<cplx> &x);
};

/// @brief Factor \f$s I - A\f$, reusing the analysis.
[[nodiscard]] sparse_shifted_lu shift(const sparse_resolvent &R, cplx s);

/// @brief Shifted solve: \f$x \leftarrow (s I - A)^{-1} b\f$. `x` may alias `b`.
void solve(const sparse_shifted_lu &F, const vec<cplx> &b, vec<cplx> &x);

} // namespace num
