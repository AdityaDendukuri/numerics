/// @file algebra/concepts.hpp
/// @brief Algebraic structure: what an object *is*, independent of how it is stored.
///
/// These are the mathematical contracts the rest of the library is written against.
/// They name vector spaces and linear maps over a scalar field. Storage layout is
/// described separately by the `num::repr` predicates in `container/concepts.hpp`.
#pragma once

#include "algebra/ops.hpp"
#include "algebra/scalar.hpp"
#include "core/types.hpp"
#include <concepts>
#include <type_traits>

namespace num {

// =============================================================================
// 1. Algebraic structure
// =============================================================================

using scalars::Field;

/// @brief Scalar field \f$\mathbb{K}\f$ over which the containers below are defined.
template <typename T>
concept Scalar = Field<T>;

/// @brief Abelian group under addition: closed, with an identity and inverses.
///
/// Closure is the substantive requirement and the one an indexing-only concept
/// misses: `V(n)` must construct an element, and coordinates must be writable, or
/// there is nowhere for a sum to live.
template <class V, class T = scalar_t<V>>
concept AdditiveGroup = scalars::Field<T> && std::copy_constructible<V> &&
    requires(V &v, const V &cv, idx n, T a) {
    V(n);
    { cv.size() } -> std::convertible_to<idx>;
    { cv[idx{0}] } -> std::convertible_to<T>;
    { v[idx{0}] = a };
};

/// @brief Vector space over a field \f$\mathbb{K}\f$: an abelian group with a compatible scalar action.
template <class V, class T = scalar_t<V>>
concept VectorSpace = AdditiveGroup<V, T> && requires(V &v, const V &x, T a) {
    algebra::scale_inplace(v, a);
    algebra::axpy_into(a, x, v);
};

/// @brief Inner product space: a vector space carrying \f$\langle \cdot,\cdot \rangle\f$.
template <class V, class T = scalar_t<V>>
concept InnerProductSpace = VectorSpace<V, T> && requires(const V &x, const V &y) {
    { algebra::inner(x, y) } -> std::convertible_to<T>;
};

/// @brief Normed space: a vector space carrying \f$\|\cdot\|\f$.
template <class V, class T = scalar_t<V>>
concept NormedSpace = VectorSpace<V, T> && requires(const V &x) {
    { algebra::norm_of(x) } -> std::convertible_to<scalars::real_t<T>>;
};

/// @brief Hilbert space: complete inner product space, with \f$\|x\|^2 = \langle x,x \rangle\f$.
///
/// Completeness is automatic in finite dimension, so what distinguishes this from
/// an inner product space in practice is that the norm is the induced one, the
/// compatibility law every Krylov method assumes.
template <class V, class T = scalar_t<V>>
concept HilbertSpace = InnerProductSpace<V, T> && NormedSpace<V, T>;

/// @brief Vector space whose coordinates can be written.
template <class V, class T = scalar_t<V>>
concept MutableVectorSpace = VectorSpace<V, T> && requires(V &v, T x) {
    { v[idx{0}] = x };
};

/// @brief Scalar function \f$f: \mathbb{K} \to \mathbb{K}\f$ over a field.
///
/// Supplied as a callable, so a lambda, a function pointer, and a `std::function`
/// are equally admissible. The result must lie in the same field as the argument,
/// which is what lets a quadrature rule accumulate into it.
template <class F, class T = real>
concept ScalarFunction = scalars::Field<T> && std::invocable<F, T> && requires(F f, T x) {
    { f(x) } -> std::convertible_to<T>;
};

/// @brief Scalar function supplied together with its derivative.
template <class F, class D, class T = real>
concept DifferentiableFunction = ScalarFunction<F, T> && ScalarFunction<D, T>;

/// @brief Linear map \f$A: \mathbb{K}^{n} \to \mathbb{K}^{m}\f$ presented entrywise.
template <class A, class T = entry_t<A>>
concept MatrixSpace = Field<T> && requires(const A &a) {
    { a.rows() } -> std::convertible_to<idx>;
    { a.cols() } -> std::convertible_to<idx>;
    { a(idx{0}, idx{0}) } -> std::convertible_to<T>;
};

/// @brief Entrywise-presented linear map whose entries can be written.
template <class A, class T = entry_t<A>>
concept MutableMatrixSpace = MatrixSpace<A, T> && requires(A &a, T x) {
    { a(idx{0}, idx{0}) = x };
};



} // namespace num
