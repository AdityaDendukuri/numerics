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

/// @brief Scalar field \f$\mathbb{K}\f$ over which vector spaces and containers are defined.
///
/// Satisfies the field axioms: closed under addition, subtraction, multiplication, and
/// division (by non-zero elements), possessing an additive identity (0) and multiplicative
/// identity (1), and providing a well-defined real modulus \f$|x|\f$.
///
/// ### Supported Field Types
/// * `double`, `float` (Real field \f$\mathbb{R}\f$)
/// * `std::complex<double>`, `std::complex<float>` (Complex field \f$\mathbb{C}\f$)
///
/// ### Example
/// @code
/// static_assert(num::Scalar<double>);
/// static_assert(num::Scalar<std::complex<double>>);
/// static_assert(!num::Scalar<int>); // Integers form a ring, not a field
/// @endcode
///
/// @tparam T The candidate scalar type.
/// @see num::scalars::Field
template <typename T>
concept Scalar = Field<T>;

/// @brief Abelian group under addition: closed, with identity 0 and additive inverses.
///
/// Closure is the fundamental algebraic requirement: elements of type `V` must be constructible
/// with a specified dimension `V(n)`, have a queryable size, and expose readable/writable coordinates.
///
/// ### Axioms
/// 1. **Closure & Associativity:** \f$(u + v) + w = u + (v + w) \in V\f$
/// 2. **Identity Element:** \f$v + 0 = v\f$
/// 3. **Inverse Element:** \f$v + (-v) = 0\f$
/// 4. **Commutativity:** \f$u + v = v + u\f$
///
/// ### Supported Models
/// * `num::Vector`, `num::CVector`
/// * `std::vector<double>`, `std::vector<float>`
///
/// @tparam V Container or vector type.
/// @tparam T Underlying scalar type (defaults to `scalar_t<V>`).
/// @see num::VectorSpace
template <class V, class T = scalar_t<V>>
concept AdditiveGroup = scalars::Field<T> && std::copy_constructible<V> &&
    requires(V &v, const V &cv, idx n, T a) {
    V(n);
    { cv.size() } -> std::convertible_to<idx>;
    { cv[idx{0}] } -> std::convertible_to<T>;
    { v[idx{0}] = a };
};

/// @brief Vector space over a field \f$\mathbb{K}\f$: an abelian group with a compatible scalar action.
///
/// Extends `AdditiveGroup` with compatible scalar multiplication and linear combination (axpy):
///
/// ### Axioms
/// 1. **Distributivity over Vectors:** \f$a (u + v) = a u + a v\f$
/// 2. **Distributivity over Scalars:** \f$(a + b) v = a v + b v\f$
/// 3. **Compatibility:** \f$(a b) v = a (b v)\f$
/// 4. **Identity Scalar:** \f$1 \cdot v = v\f$
///
/// ### Example
/// @code
/// static_assert(num::VectorSpace<num::Vector>);
/// static_assert(num::VectorSpace<std::vector<float>>);
/// static_assert(!num::VectorSpace<std::span<const double>>); // Non-owning views cannot receive sums
/// @endcode
///
/// @tparam V Container or vector space type.
/// @tparam T Scalar field type.
/// @see num::InnerProductSpace, num::NormedSpace
template <class V, class T = scalar_t<V>>
concept VectorSpace = AdditiveGroup<V, T> && requires(V &v, const V &x, T a) {
    algebra::scale_inplace(v, a);
    algebra::axpy_into(a, x, v);
};

/// @brief Inner product space: a vector space carrying a bilinear/sesquilinear form \f$\langle \cdot,\cdot \rangle\f$.
///
/// Provides the `algebra::inner(x, y)` operation satisfying:
/// 1. **Conjugate Symmetry:** \f$\langle x, y \rangle = \overline{\langle y, x \rangle}\f$
/// 2. **Linearity in Second Argument:** \f$\langle x, a y + b z \rangle = a \langle x, y \rangle + b \langle x, z \rangle\f$
/// 3. **Positive-Definiteness:** \f$\langle x, x \rangle > 0\f$ for all \f$x \ne 0\f$
///
/// @tparam V Vector space type.
/// @tparam T Scalar field type.
/// @see num::HilbertSpace
template <class V, class T = scalar_t<V>>
concept InnerProductSpace = VectorSpace<V, T> && requires(const V &x, const V &y) {
    { algebra::inner(x, y) } -> std::convertible_to<T>;
};

/// @brief Normed space: a vector space carrying a norm \f$\|\cdot\|\f$.
///
/// Provides `algebra::norm_of(x)` satisfying:
/// 1. **Non-negativity:** \f$\|x\| \ge 0\f$, and \f$\|x\| = 0 \iff x = 0\f$
/// 2. **Absolute Homogeneity:** \f$\|a x\| = |a| \|x\|\f$
/// 3. **Triangle Inequality:** \f$\|x + y\| \le \|x\| + \|y\|\f$
///
/// @tparam V Vector space type.
/// @tparam T Scalar field type.
/// @see num::HilbertSpace
template <class V, class T = scalar_t<V>>
concept NormedSpace = VectorSpace<V, T> && requires(const V &x) {
    { algebra::norm_of(x) } -> std::convertible_to<scalars::real_t<T>>;
};

/// @brief Hilbert space: complete inner product space whose norm is induced by the inner product.
///
/// Conjoins `InnerProductSpace` and `NormedSpace` with the compatibility law:
/// \f[
/// \|x\|^2 = \langle x, x \rangle
/// \f]
/// Every Krylov solver (CG, GMRES, MINRES) fundamentally operates over a Hilbert space.
///
/// @tparam V Vector space type.
/// @tparam T Scalar field type.
template <class V, class T = scalar_t<V>>
concept HilbertSpace = InnerProductSpace<V, T> && NormedSpace<V, T>;

/// @brief Vector space whose individual coordinate entries can be directly assigned.
///
/// @tparam V Vector space type.
/// @tparam T Scalar field type.
template <class V, class T = scalar_t<V>>
concept MutableVectorSpace = VectorSpace<V, T> && requires(V &v, T x) {
    { v[idx{0}] = x };
};

/// @brief Scalar function \f$f: \mathbb{K} \to \mathbb{K}\f$ over a field.
///
/// Accepts callables (lambdas, function pointers, `std::function`) mapping a scalar to a scalar.
///
/// @tparam F Callable object type.
/// @tparam T Scalar field type (defaults to `real`).
template <class F, class T = real>
concept ScalarFunction = scalars::Field<T> && std::invocable<F, T> && requires(F f, T x) {
    { f(x) } -> std::convertible_to<T>;
};

/// @brief Scalar function supplied together with its exact derivative \f$(f, f')\f$.
///
/// Used in root-finding algorithms such as Newton–Raphson and Halley's method.
///
/// @tparam F Callable function type.
/// @tparam D Callable derivative type.
/// @tparam T Scalar field type.
template <class F, class D, class T = real>
concept DifferentiableFunction = ScalarFunction<F, T> && ScalarFunction<D, T>;

/// @brief Linear map \f$A: \mathbb{K}^{n} \to \mathbb{K}^{m}\f$ presented as an entrywise 2D indexable space.
///
/// @tparam A Matrix or 2D array type.
/// @tparam T Entry scalar type.
template <class A, class T = entry_t<A>>
concept MatrixSpace = Field<T> && requires(const A &a) {
    { a.rows() } -> std::convertible_to<idx>;
    { a.cols() } -> std::convertible_to<idx>;
    { a(idx{0}, idx{0}) } -> std::convertible_to<T>;
};

/// @brief Entrywise-presented linear map whose entries can be mutated.
///
/// @tparam A Matrix type.
/// @tparam T Entry scalar type.
template <class A, class T = entry_t<A>>
concept MutableMatrixSpace = MatrixSpace<A, T> && requires(A &a, T x) {
    { a(idx{0}, idx{0}) = x };
};



} // namespace num
