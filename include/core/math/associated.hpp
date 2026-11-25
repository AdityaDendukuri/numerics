/// @file core/math/associated.hpp
/// @brief Associated types: scalar, real part, domain and codomain.
#pragma once

#include <complex>
#include <cstddef>
#include <type_traits>
#include <utility>

namespace num::scalars {

/// @brief Whether `T` is a `std::complex`.
template <class T>
struct is_complex : std::false_type {};

template <class T>
struct is_complex<std::complex<T>> : std::true_type {};

template <class T>
inline constexpr bool is_complex_v = is_complex<std::remove_cvref_t<T>>::value;

/// @brief Real field of `T`: `U` for `complex<U>`, else `U`.
template <class T>
struct real_of {
    using type = std::remove_cvref_t<T>;
};

template <class T>
struct real_of<std::complex<T>> {
    using type = T;
};

/// @brief Real field of scalar `T`.
template <class T>
using real_t = typename real_of<std::remove_cvref_t<T>>::type;

} // namespace num::scalars

namespace num::math {

namespace detail {

template <class T>
concept has_value_type = requires { typename T::value_type; };

template <class T>
struct scalar_of {
    using type = void;
};

template <class T>
requires has_value_type<T>
struct scalar_of<T> {
    using type = typename T::value_type;
};

template <class T>
requires(!has_value_type<T>) && requires(const T &v) { v[std::size_t{0}]; }
struct scalar_of<T> {
    using type = std::remove_cvref_t<decltype(std::declval<const T &>()[std::size_t{0}])>;
};

template <class T>
struct domain_of {
    using type = void;
};

template <class T>
requires requires { typename T::domain_type; }
struct domain_of<T> {
    using type = typename T::domain_type;
};

template <class T>
struct codomain_of {
    using type = void;
};

template <class T>
requires requires { typename T::codomain_type; }
struct codomain_of<T> {
    using type = typename T::codomain_type;
};

} // namespace detail

/// @brief Scalar type of a container: its `value_type` or the type of `v[i]`, else `void`.
template <class T>
using scalar_t = typename detail::scalar_of<std::remove_cvref_t<T>>::type;

/// @brief Domain of an operator \f$A: X \to Y\f$: the space \f$X\f$.
template <class T>
using domain_t = typename detail::domain_of<std::remove_cvref_t<T>>::type;

/// @brief Codomain of an operator \f$A: X \to Y\f$: the space \f$Y\f$.
template <class T>
using codomain_t = typename detail::codomain_of<std::remove_cvref_t<T>>::type;

} // namespace num::math
