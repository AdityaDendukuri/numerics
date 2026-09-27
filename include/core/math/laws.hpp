/// @file laws.hpp
/// @brief The laws an operator can claim, and `claims`, the one way to read them.
///
/// A law is a property that an algorithm relies on and the compiler cannot decide. A type
/// declares the laws it satisfies as `using laws = num::law::list<num::law::spd>;`, or the
/// caller attaches one with `num::assume<L>(A)`, which samples it at runtime. A law derives
/// from the laws it implies, so an SPD operator also claims `psd` and `self_adjoint`.
///
/// ```
/// diagonally_dominant             Jacobi, Gauss-Seidel, LU without pivoting
/// self_adjoint                    MINRES, Lanczos, eig_sym
///  └ psd                          graph Laplacians
///     └ spd                       CG, PCG, Cholesky
/// self_adjoint_on<S> └ psd_on<S> └ spd_on<S>    the same laws restricted to a subspace S
/// ```
#pragma once

#include <concepts>
#include <type_traits>

namespace num::law {

/// @brief The laws a type declares.
template <class... Laws>
struct list {};

/// @brief \f$|a_{ii}| > \sum_{j \neq i} |a_{ij}|\f$ in every row. Incomparable with the
/// self-adjoint laws.
struct diagonally_dominant {};

/// @brief \f$A = A^*\f$: symmetric over \f$\mathbb{R}\f$, Hermitian over \f$\mathbb{C}\f$.
struct self_adjoint {};

/// @brief \f$\langle x, Ax \rangle \geq 0\f$. Admits a null space.
struct psd : self_adjoint {};

/// @brief \f$\langle x, Ax \rangle > 0\f$ for \f$x \neq 0\f$: invertible, Cholesky-factorable.
struct spd : psd {};

/// @brief The restriction of A to the subspace S is self-adjoint and maps S into itself.
template <class S>
struct self_adjoint_on {};

/// @brief The restriction of A to S is positive semidefinite.
template <class S>
struct psd_on : self_adjoint_on<S> {};

/// @brief The restriction of A to S is positive definite, as PCG on S needs.
template <class S>
struct spd_on : psd_on<S> {};

namespace detail {

template <class T>
struct declared {
    using type = list<>;
};

template <class T>
requires requires { typename T::laws; }
struct declared<T> {
    using type = typename T::laws;
};

template <class L, class... Declared>
consteval bool implied(list<Declared...>) {
    return (std::derived_from<Declared, L> || ...);
}

} // namespace detail

/// @brief The laws `T` declares.
template <class T>
using declared_t = typename detail::declared<std::remove_cvref_t<T>>::type;

} // namespace num::law

namespace num {

/// @brief True when `T` declares `L`, or a law that implies it.
template <class T, class L>
concept claims = law::detail::implied<L>(law::declared_t<T>{});

} // namespace num
