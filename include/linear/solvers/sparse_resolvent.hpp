/// @file sparse_resolvent.hpp
/// @brief Shifted sparse solves through an available direct backend.
#pragma once

#include "container/vector.hpp"
#include "core/types.hpp"
#include "linear/solve.hpp"
#include "linear/sparse/sparse.hpp"
#include <memory>
#include <vector>

namespace num {

/// True when numerics was built with SuiteSparse KLU, which factors the shifted systems.
[[nodiscard]] bool sparse_resolvent_available() noexcept;

class sparse_shifted_lu;

/// @brief The sparsity of \f$sI - A\f$ analyzed once, ready to factor it for any shift.
///
/// Factored by KLU's complex interface: the symbolic analysis (block triangular form and AMD
/// ordering) is computed once, and each `shift` does only the numeric factorization. `shift` may
/// be called concurrently on one analysis, so independent shifts (Talbot nodes, say) factor in
/// parallel; one factor serves one thread at a time. Without KLU, `shift` throws rather than
/// silently densifying a large matrix. Copies share the analysis.
class sparse_resolvent {
  public:
    /// Analyze A's sparsity pattern. A must store its diagonal explicitly.
    explicit sparse_resolvent(const spmat &A);

    [[nodiscard]] idx size() const noexcept;

    struct analysis;

  private:
    std::shared_ptr<const analysis> analysis_;
    friend class sparse_shifted_lu;
    friend sparse_shifted_lu shift(const sparse_resolvent &R, cplx s);
};

/// @brief \f$sI - A\f$ for one shift \f$s\f$, factored by the available sparse backend.
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

/// @brief Factor \f$sI - A\f$ numerically, reusing the symbolic analysis.
[[nodiscard]] sparse_shifted_lu shift(const sparse_resolvent &R, cplx s);

/// @brief Solve \f$(sI - A)x = b\f$. `x` may be `b`.
void solve(const sparse_shifted_lu &F, const vec<cplx> &b, vec<cplx> &x);

} // namespace num
