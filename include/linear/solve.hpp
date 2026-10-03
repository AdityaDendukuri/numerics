/// @file linear/solve.hpp
/// @brief The solve protocol every factorization follows.
///
/// A factorization is a value, and solving with it is one free function chosen by its type:
///
///     auto F = lu(A);            // or cholesky(...), lu(band), lu(R, blocks(levels)), ...
///     solve(F, b, x);            // x = A^{-1} b; x may be b
///     x = solve(F, b);           // the same, allocating x
///     solve(transpose(F), b, x); // x = A^{-T} b
///
/// Each factorization type supplies `solve(F, b, x)`, and `solve_transpose(F, b, x)` when it
/// can, for `vec<real>` and `mat<real>` right-hand sides. Everything else here is written once
/// on top of those. Overloads resolve at compile time, so a solve costs no indirect call.
#pragma once

#include "container/matrix.hpp"
#include "container/vector.hpp"
#include <type_traits>

namespace num {

/// @brief A factorization of a square \f$A\f$ that applies \f$A^{-1}\f$ and \f$A^{-T}\f$.
///
/// The out-parameter forms may alias their input, so a caller can solve in place.
template <class F>
concept factorization =
    requires(const F &factor, const vec<real> &v, const mat<real> &m, vec<real> &vout,
             mat<real> &mout) {
        solve(factor, v, vout);
        solve(factor, m, mout);
        solve_transpose(factor, v, vout);
        solve_transpose(factor, m, mout);
    };

/// @brief Solve \f$Ax = b\f$, allocating `x`. Prefer the out-parameter form in hot loops.
template <class F, class RHS>
[[nodiscard]] RHS solve(const F &factor, const RHS &b) requires requires(RHS &x) {
    solve(factor, b, x);
}
{
    RHS x;
    solve(factor, b, x);
    return x;
}

/// @brief Solve \f$A^Tx = b\f$, allocating `x`.
template <class F, class RHS>
[[nodiscard]] RHS solve_transpose(const F &factor, const RHS &b) requires requires(RHS &x) {
    solve_transpose(factor, b, x);
}
{
    RHS x;
    solve_transpose(factor, b, x);
    return x;
}

namespace detail {

template <class F>
struct transposed_factor {
    const F &base;
};

} // namespace detail

/// @brief View a factorization of \f$A\f$ as one of \f$A^T\f$, so `solve` covers both.
///
/// The view stores only a reference: it neither transposes nor copies the factors, and it
/// must not outlive them.
template <class F>
requires requires(const F &factor, const vec<real> &b, vec<real> &x) {
    solve_transpose(factor, b, x);
}
[[nodiscard]] inline detail::transposed_factor<F> transpose(const F &factor) {
    return {factor};
}

template <class F>
requires(!std::is_lvalue_reference_v<F> &&
         requires(const std::remove_reference_t<F> &factor, const vec<real> &b, vec<real> &x) {
             solve_transpose(factor, b, x);
         }) detail::transposed_factor<std::remove_reference_t<F>> transpose(F &&) = delete;

template <class F>
[[nodiscard]] inline const F &transpose(detail::transposed_factor<F> factor) {
    return factor.base;
}

template <class F, class RHS>
inline void solve(detail::transposed_factor<F> factor, const RHS &b, RHS &x) {
    solve_transpose(factor.base, b, x);
}

template <class F, class RHS>
inline void solve_transpose(detail::transposed_factor<F> factor, const RHS &b, RHS &x) {
    solve(factor.base, b, x);
}

} // namespace num
