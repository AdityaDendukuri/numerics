/// @file core/math/laws.hpp
/// @brief Operator laws: declared properties an algorithm relies on, read with `claims`.
#pragma once

#include <concepts>
#include <type_traits>

namespace num::law {

/// @brief List of declared laws.
template <class... Laws>
struct list {};

/// @brief Diagonal dominance: \f$|a_{ii}| > \sum_{j \ne i} |a_{ij}|\f$.
struct diagonally_dominant {};

/// @brief Self-adjoint: \f$A = A^*\f$.
struct self_adjoint {};

/// @brief Positive semidefinite: \f$\langle x, A x \rangle \ge 0\f$.
struct psd : self_adjoint {};

/// @brief Positive definite: \f$\langle x, A x \rangle > 0\f$ for \f$x \ne 0\f$.
struct spd : psd {};

/// @brief Self-adjoint on subspace \f$S\f$: \f$A S \subseteq S\f$ and \f$A|_S = A|_S^*\f$.
template <class S>
struct self_adjoint_on {};

/// @brief Positive semidefinite on subspace \f$S\f$.
template <class S>
struct psd_on : self_adjoint_on<S> {};

/// @brief Positive definite on subspace \f$S\f$.
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

/// @brief Laws declared by `T`.
template <class T>
using declared_t = typename detail::declared<std::remove_cvref_t<T>>::type;

} // namespace num::law

namespace num {

/// @brief Whether `T` declares law `L` or one implying it.
template <class T, class L>
concept claims = law::detail::implied<L>(law::declared_t<T>{});

} // namespace num
