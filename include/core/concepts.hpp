/// @file core/concepts.hpp
/// @brief Storage-shaped concepts shared by numerical modules.
#pragma once

#include "core/types.hpp"
#include "core/vector.hpp"
#include <concepts>

namespace num {

/// @brief Indexed real-valued vector interface.
template <class V>
concept VectorLike = requires(V v, const V cv, idx i) {
    {cv.size()}->std::convertible_to<idx>;
    {cv[i]}->std::convertible_to<real>;
    {v[i]}->std::convertible_to<real>;
};

/// @brief Mutable indexed real-valued vector interface.
template <class V>
concept MutableVectorLike = VectorLike<V> && requires(V v, idx i, real a) {
    {v[i] = a}->std::same_as<real &>;
};

/// @brief Contiguous real-valued vector storage.
template <class V>
concept ContiguousVectorLike = VectorLike<V> && requires(V v, const V cv) {
    {cv.data()}->std::convertible_to<const real *>;
    {v.data()}->std::convertible_to<real *>;
};

/// @brief Dense row-major real matrix interface.
template <class A>
concept DenseMatrixLike = requires(A a, const A ca, idx i, idx j) {
    {ca.rows()}->std::convertible_to<idx>;
    {ca.cols()}->std::convertible_to<idx>;
    {ca(i, j)}->std::convertible_to<real>;
    {a(i, j)}->std::convertible_to<real>;
};

/// @brief Mutable dense row-major real matrix interface.
template <class A>
concept MutableDenseMatrixLike = DenseMatrixLike<A> && requires(A a, idx i, idx j, real x) {
    {a(i, j) = x}->std::same_as<real &>;
};

/// @brief Contiguous dense row-major real matrix storage.
template <class A>
concept ContiguousDenseMatrixLike = DenseMatrixLike<A> && requires(A a, const A ca) {
    {ca.data()}->std::convertible_to<const real *>;
    {a.data()}->std::convertible_to<real *>;
};

} // namespace num
