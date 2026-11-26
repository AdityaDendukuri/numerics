/// @file linear/solve.hpp
/// @brief Solve protocol: `solve(F, b, x)` and `solve_transpose(F, b, x)` for every factorization.
#pragma once

#include "container/matrix.hpp"
#include "container/vector.hpp"
#include <type_traits>

namespace num {

/// @brief Factorization that solves with \f$A\f$ and \f$A^T\f$; the out-parameter forms may alias.
template <class F>
concept factorization =
    requires(const F &factor, const vec<real> &v, const mat<real> &m, vec<real> &vout,
             mat<real> &mout) {
        solve(factor, v, vout);
        solve(factor, m, mout);
        solve_transpose(factor, v, vout);
        solve_transpose(factor, m, mout);
    };

/// @brief Solve with a factorization: \f$x = A^{-1} b\f$. Allocates.
template <class F, class RHS>
[[nodiscard]] RHS solve(const F &factor, const RHS &b) requires requires(RHS &x) {
    solve(factor, b, x);
}
{
    RHS x;
    solve(factor, b, x);
    return x;
}

/// @brief Solve the transposed system with a factorization: \f$x = A^{-T} b\f$. Allocates.
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

/// @brief View a factorization of \f$A\f$ as one of \f$A^T\f$. References `F`.
/// @warning References `F`; it must outlive the view.
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
