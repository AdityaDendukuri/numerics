/// @file algebra/concepts.hpp
/// @brief The public algebraic vocabulary, and the entry points into the hierarchy.
///
/// The hierarchy is defined once in `core/math/concepts.hpp` and re-exported into `num`. It
/// covers `field`, `additive_group`, `vector_space`, `normed_space`, `inner_product_space`,
/// `hilbert_space`, and the operator concepts up to `spd_operator`. This header used to define
/// a second, duck-typed copy of the first five. That copy is gone, and with it the class
/// of bug in which `num::vector_space` and `num::math::vector_space` were different
/// predicates sharing one name.
///
/// What remains here is what genuinely sits outside the hierarchy: entrywise matrix access,
/// scalar functions, and the opt-in that admits foreign containers.
///
/// Storage layout is described separately by the `num::repr` predicates in
/// `container/concepts.hpp`. Bandedness is a statement about memory, not about a map.
#pragma once

#include "algebra/ops.hpp"
#include "algebra/scalar.hpp"
#include "core/math/concepts.hpp"
#include "core/types.hpp"
#include <complex>
#include <concepts>
#include <type_traits>
#include <vector>

namespace num::math {

/// @brief `std::vector` of a floating-point type is a Hilbert space, and says so.
///
/// The hierarchy asks a type to claim its laws rather than inferring them from syntax, which
/// prevents `std::string`, which defines `operator+`, from being mistaken for a vector
/// space. A standard container cannot declare a member typedef, so the claim is attached
/// from outside here. `std::vector<int>` is not covered. The integers form a ring rather
/// than a field, so it has no level on this hierarchy.
template <std::floating_point T>
struct claims_of<std::vector<T>> {
    using type = type_list<law::hilbert_space>;
};

template <std::floating_point T>
struct claims_of<std::vector<std::complex<T>>> {
    using type = type_list<law::hilbert_space>;
};

} // namespace num::math

namespace num {

/// @brief scalar field \f$\mathbb{K}\f$ over which spaces and containers are defined.
///
/// Satisfies the field axioms: closed under `+ - * /` (by non-zero elements), with
/// additive identity 0, multiplicative identity 1, and a well-defined modulus \f$|x|\f$.
///
/// ### Supported
/// * `double` and `float`, the real field \f$\mathbb{R}\f$
/// * `std::complex<double>` and `std::complex<float>`, the complex field \f$\mathbb{C}\f$
///
/// ### Example
/// @code
/// static_assert(num::scalar<double>);
/// static_assert(num::scalar<std::complex<double>>);
/// static_assert(!num::scalar<int>); // integers form a ring, not a field
/// @endcode
///
/// @tparam T The candidate scalar type.
///
/// A synonym for `num::field`, kept because "scalar" is what the surrounding mathematics
/// calls it. Both name the one certified definition in `core/math/concepts.hpp`.
template <typename T>
concept scalar = field<T>;

/// @brief Linear map presented as an entrywise-indexable 2D array.
///
/// Distinct from `linear_operator`, which can only be applied. A matrix-free operator is a
/// `linear_operator` and not a `matrix_space`. A dense matrix is both. A routine that reads
/// individual entries, such as pivoting or banded factorization, requires `matrix_space`.
///
/// @tparam A mat or 2D array type.
/// @tparam T Entry scalar type.
template <class A, class T = entry_t<A>>
concept matrix_space = field<T> && requires(const A &a) {
    { a.rows() } -> std::convertible_to<idx>;
    { a.cols() } -> std::convertible_to<idx>;
    { a(idx{0}, idx{0}) } -> std::convertible_to<T>;
};

/// @brief Entrywise-presented linear map whose entries can be mutated.
///
/// @tparam A mat type.
/// @tparam T Entry scalar type.
template <class A, class T = entry_t<A>>
concept mutable_matrix_space = matrix_space<A, T> && requires(A &a, T x) {
    { a(idx{0}, idx{0}) = x };
};

/// @brief scalar function \f$f: \mathbb{K} \to \mathbb{K}\f$ over a field.
///
/// Accepts any callable mapping a scalar to a scalar, including lambdas, function
/// pointers and `std::function`.
///
/// @tparam F Callable object type.
/// @tparam T scalar field type (defaults to `real`).
template <class F, class T = real>
concept scalar_function = field<T> && std::invocable<F, T> && requires(F f, T x) {
    { f(x) } -> std::convertible_to<T>;
};

/// @brief scalar function supplied with its exact derivative \f$(f, f')\f$.
///
/// Used by Newton–Raphson and Halley's method.
///
/// @tparam F Callable function type.
/// @tparam D Callable derivative type.
/// @tparam T scalar field type.
template <class F, class D, class T = real>
concept differentiable_function = scalar_function<F, T> && scalar_function<D, T>;

} // namespace num
