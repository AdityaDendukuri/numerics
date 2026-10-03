/// @file linear/solvers/hessenberg_resolvent.hpp
/// @brief O(n^2)-per-shift complex resolvent solves from one Hessenberg decomposition.
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

/// @brief \f$A\f$ reduced once to \f$A = QHQ^T\f$, ready to factor \f$sI - A\f$ for any shift.
///
/// The reduction costs \f$O(n^3)\f$ once; `shift(R, s)` then costs \f$O(n^2)\f$ per shift:
///
///     hessenberg_resolvent R(A);
///     auto F = shift(R, s);     // sI - A, factored
///     solve(F, b, x);           // x = (sI - A)^{-1} b, as many times as needed
///
/// Copies share the reduction, and each shifted factor keeps it alive.
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

/// @brief \f$sI - A\f$ for one shift \f$s\f$, factored as \f$sI - H\f$ in Hessenberg coordinates.
struct hessenberg_shifted_lu {
    std::shared_ptr<const hessenberg_decomposition> decomp;
    cplx s;
    array<cplx> factor;
    array<idx> pivots;

    [[nodiscard]] idx size() const noexcept { return decomp->size(); }
};

/// @brief Factor \f$sI - A\f$ in \f$O(n^2)\f$.
[[nodiscard]] hessenberg_shifted_lu shift(const hessenberg_resolvent &R, cplx s);

/// @brief Solve \f$(sI - A)x = b\f$. `x` may be `b`.
void solve(const hessenberg_shifted_lu &F, const vec<cplx> &b, vec<cplx> &x);

/// @brief Solve \f$(sI - A)x = b\f$ for a real \f$b\f$; the solution is complex.
void solve(const hessenberg_shifted_lu &F, const vec<real> &b, vec<cplx> &x);
[[nodiscard]] vec<cplx> solve(const hessenberg_shifted_lu &F, const vec<real> &b);

/// @brief Solve one right-hand side at many shifts, in parallel over the shifts.
///
/// The same as `solve(shift(R, s), b)` for each `s`, with the projection of `b` shared and
/// one factor buffer per thread.
[[nodiscard]] array<vec<cplx>> solve_batch(const hessenberg_resolvent &R,
                                           const array<cplx> &shifts, const vec<real> &b);

/// @brief Solve several right-hand sides at many shifts: `result[shift][rhs]`.
[[nodiscard]] array<array<vec<cplx>>> solve_batch(const hessenberg_resolvent &R,
                                                  const array<cplx> &shifts,
                                                  const array<vec<real>> &rhs_list);

} // namespace num
