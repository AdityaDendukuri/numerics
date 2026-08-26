/// @file models.hpp
/// @brief Explicit certification that a C++ type models mathematical laws.
#pragma once

#include <complex>
#include <concepts>
#include <type_traits>

namespace num::math {

template <class... Ts>
struct type_list {};

namespace law {

struct semiring {};
struct ring : semiring {};
struct field : ring {};

struct additive_group {};
struct vector_space : additive_group {};
struct normed_space : vector_space {};
struct inner_product_space : normed_space {};

struct map {};
struct linear_map : map {};
struct linear_subspace {};

} // namespace law

/// Customization point for the strongest independent laws modeled by T.
template <class T>
struct model_of {
    using laws = type_list<>;
};

template <std::floating_point T>
struct model_of<T> {
    using laws = type_list<law::field>;
};

template <std::floating_point T>
struct model_of<std::complex<T>> {
    using laws = type_list<law::field>;
};

namespace detail {

template <class Law, class... Models>
inline constexpr bool list_models_v = (std::derived_from<Models, Law> || ...);

template <class Law, class... Models>
consteval bool list_models(type_list<Models...>) {
    return list_models_v<Law, Models...>;
}

} // namespace detail

/// True when T explicitly certifies Law or a stronger law that implies it.
template <class T, class Law>
concept Models = detail::list_models<Law>(typename model_of<std::remove_cvref_t<T>>::laws{});

} // namespace num::math
