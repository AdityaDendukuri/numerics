/// @file concepts.hpp
/// @brief Mathematical concepts combining executable protocols with certified laws.
#pragma once

#include "core/math/associated.hpp"
#include "core/math/models.hpp"
#include "core/math/operations.hpp"
#include <concepts>
#include <type_traits>

namespace num::math {

template <class T>
concept Field = Models<T, law::field> && requires(T a, T b) {
    {a + b}->std::convertible_to<T>;
    {a - b}->std::convertible_to<T>;
    {a * b}->std::convertible_to<T>;
    {a / b}->std::convertible_to<T>;
    T{0};
    T{1};
};

template <class V>
concept VectorProtocol =
    Field<scalar_t<V>> && std::copy_constructible<V> && requires(V & v, const V &x, scalar_t<V> a) {
    {dimension(x)}->std::integral;
    {zero_like(x)}->std::same_as<V>;
    scale(a, v);
    axpy(a, x, v);
};

template <class V>
concept VectorSpace = VectorProtocol<V> && Models<V, law::vector_space>;

template <class V>
concept NormedSpace = VectorSpace<V> && Models<V, law::normed_space> && requires(const V &v) {
    norm(v);
};

template <class V>
concept InnerProductSpace =
    NormedSpace<V> && Models<V, law::inner_product_space> && requires(const V &x, const V &y) {
    {inner(x, y)}->std::convertible_to<scalar_t<V>>;
};

template <class Op>
concept LinearMap =
    Models<Op, law::linear_map> && VectorSpace<domain_t<Op>> && VectorSpace<codomain_t<Op>>;

template <class Op>
concept LinearOperator =
    LinearMap<Op> && requires(const Op &op, const domain_t<Op> &x, codomain_t<Op> &y) {
    apply(op, x, y);
    {op.rows()}->std::integral;
    {op.cols()}->std::integral;
};

template <class Op, class V>
concept EndomorphismOn =
    LinearOperator<Op> && std::same_as<domain_t<Op>, V> && std::same_as<codomain_t<Op>, V>;

template <class V>
concept ContiguousVector = VectorSpace<V> && requires(V & v, const V &cv) {
    {v.data()}->std::convertible_to<scalar_t<V> *>;
    {cv.data()}->std::convertible_to<const scalar_t<V> *>;
};

} // namespace num::math
