/// @file associated.hpp
/// @brief Associated mathematical types: scalar, real part, domain, and codomain.
#pragma once

#include <complex>
#include <cstddef>
#include <type_traits>
#include <utility>

namespace num::scalars {

/// @brief True when T is a std::complex specialization.
template <class T>
struct is_complex : std::false_type {};

template <class T>
struct is_complex<std::complex<T>> : std::true_type {};

template <class T>
inline constexpr bool is_complex_v = is_complex<std::remove_cvref_t<T>>::value;

/// @brief Underlying real field of T: `real_of<complex<U>>` is U, `real_of<U>` is U.
template <class T>
struct real_of {
    using type = std::remove_cvref_t<T>;
};

template <class T>
struct real_of<std::complex<T>> {
    using type = T;
};

/// @brief The real field underlying scalar T.
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

/// @brief Scalar of a container: its `value_type`, else the type of `v[i]`, else void. Void
/// lets a concept over `scalar_t` be false for an unrelated type.
template <class T>
using scalar_t = typename detail::scalar_of<std::remove_cvref_t<T>>::type;

/// @brief Space an operator acts on: its `domain_type`, or a specialization of
/// `detail::domain_of` for a type that cannot declare one.
template <class T>
using domain_t = typename detail::domain_of<std::remove_cvref_t<T>>::type;

/// @brief Space an operator maps into, found like `domain_t`.
template <class T>
using codomain_t = typename detail::codomain_of<std::remove_cvref_t<T>>::type;

} // namespace num::math
