/// @file associated.hpp
/// @brief Associated mathematical types: scalar, domain, and codomain.
#pragma once

#include <cstddef>
#include <type_traits>
#include <utility>

namespace num::math {

namespace detail {

template <class T, class = void>
struct scalar_of {
    using type = void;
};

template <class T>
struct scalar_of<T, std::void_t<typename std::remove_cvref_t<T>::value_type>> {
    using type = typename std::remove_cvref_t<T>::value_type;
};

template <class T, class = void>
struct domain_of {
    using type = void;
};

template <class T>
struct domain_of<T, std::void_t<typename std::remove_cvref_t<T>::domain_type>> {
    using type = typename std::remove_cvref_t<T>::domain_type;
};

template <class T, class = void>
struct codomain_of {
    using type = void;
};

template <class T>
struct codomain_of<T, std::void_t<typename std::remove_cvref_t<T>::codomain_type>> {
    using type = typename std::remove_cvref_t<T>::codomain_type;
};

} // namespace detail

template <class T>
using scalar_t = typename detail::scalar_of<T>::type;

template <class T>
using domain_t = typename detail::domain_of<T>::type;

template <class T>
using codomain_t = typename detail::codomain_of<T>::type;

} // namespace num::math
