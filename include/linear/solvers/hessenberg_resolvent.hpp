/// @file linear/solvers/hessenberg_resolvent.hpp
/// @brief Resolvent solves from one Hessenberg decomposition, \f$O(n^2)\f$ per shift.
#pragma once

#include "core/debug.hpp"
#include "container/matrix.hpp"
#include "core/types.hpp"
#include "container/vector.hpp"
#include "linear/factorization/hessenberg.hpp"
#include "linear/solve.hpp"
#include "linear/sparse/sparse.hpp"
#include <complex>
#include <memory>
#include <vector>

namespace num {

struct hessenberg_shifted_lu;

/// @brief Hessenberg resolvent: \f$A = Q H Q^T\f$ once, then \f$(s I - A)^{-1}\f$ per shift.
class hessenberg_resolvent {
  public:
    explicit hessenberg_resolvent(const mat<real> &A);
    explicit hessenberg_resolvent(const spmat &A);
    explicit hessenberg_resolvent(hessenberg_decomposition decomp);

    [[nodiscard]] idx size() const noexcept { return decomp_->size(); }
    [[nodiscard]] const hessenberg_decomposition &decomposition() const noexcept {
        return *decomp_;
    }

  private:
    std::shared_ptr<const hessenberg_decomposition> decomp_;
    friend hessenberg_shifted_lu shift(const hessenberg_resolvent &R, cplx s);
};

/// @brief Shifted factor: \f$s I - H\f$ in Hessenberg coordinates.
struct hessenberg_shifted_lu {
    std::shared_ptr<const hessenberg_decomposition> decomp;
    cplx s;
    array<cplx> factor;
    array<idx> pivots;

    [[nodiscard]] idx size() const noexcept { return decomp->size(); }
};

/// @brief Factor \f$s I - A\f$, \f$O(n^2)\f$.
[[nodiscard]] hessenberg_shifted_lu shift(const hessenberg_resolvent &R, cplx s);

/// @brief Shifted solve: \f$x \leftarrow (s I - A)^{-1} b\f$. `x` may alias `b`.
void solve(const hessenberg_shifted_lu &F, const vec<cplx> &b, vec<cplx> &x);

/// @brief Shifted solve, real right-hand side: \f$x \leftarrow (s I - A)^{-1} b\f$.
void solve(const hessenberg_shifted_lu &F, const vec<real> &b, vec<cplx> &x);
[[nodiscard]] vec<cplx> solve(const hessenberg_shifted_lu &F, const vec<real> &b);

/// @brief Batched shifted solve: \f$(s_k I - A)^{-1} b\f$ for each shift, in parallel.
[[nodiscard]] array<vec<cplx>> solve_batch(const hessenberg_resolvent &R,
                                           const array<cplx> &shifts, const vec<real> &b);

/// @brief Batched shifted solve: \f$(s_k I - A)^{-1} b_j\f$, indexed `[shift][rhs]`.
[[nodiscard]] array<array<vec<cplx>>> solve_batch(const hessenberg_resolvent &R,
                                                  const array<cplx> &shifts,
                                                  const array<vec<real>> &rhs_list);

} // namespace num
