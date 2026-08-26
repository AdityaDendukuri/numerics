/// @file operator/concepts.hpp
/// @brief Structural and axiomatic contracts for linear and nonlinear operators.
///
/// The contracts here come in two kinds and are deliberately built differently.
///
/// `num::LinearOperator` is **structure**: an object that maps a vector of one
/// space into another and knows its dimensions. The compiler decides it outright.
///
/// Everything below it is an **axiom** — self-adjointness, definiteness, unitarity.
/// No type can prove those, so each is a claim the caller attaches with
/// `num::operators::assume_*`, recorded in the `num::property` lattice and sampled
/// at runtime under the active diagnostic preset.
///
/// The named concepts form a subsumption chain (each is a conjunction containing
/// the next weaker one) so that overloads order correctly: an SPD operator passed
/// to a set containing CG, MINRES and GMRES selects CG, not an ambiguity.
#pragma once

#include "container/concepts.hpp"
#include "container/vector.hpp"
#include "algebra/properties.hpp"
#include <type_traits>

namespace num {

// =============================================================================
// 1. Structure
// =============================================================================

/// @brief Linear map \f$A: X \to Y\f$ satisfying \f$A(\alpha x + \beta y) = \alpha A x + \beta A y\f$.
///
/// Linearity itself is an axiom, not a checkable property; what the compiler
/// verifies here is that the object presents the shape of a linear map over a
/// scalar field. `num::property::linear` carries the runtime test.
template <class Op, class X = Vector, class Y = Vector>
concept LinearOperator =
    VectorSpace<X> && MutableVectorSpace<Y> && requires(const Op &A, const X &x, Y &y) {
    { A.rows() } -> std::convertible_to<idx>;
    { A.cols() } -> std::convertible_to<idx>;
    { A.apply(x, y) };
};

/// @brief Linear map exposing its adjoint \f$A^*\f$, with \f$\langle A x, y \rangle = \langle x, A^* y \rangle\f$.
template <class Op, class X = Vector, class Y = Vector>
concept AdjointableLinearOperator =
    LinearOperator<Op, X, Y> && requires(const Op &A, const Y &y, X &x) {
    { A.apply_adjoint(y, x) };
};

// =============================================================================
// 2. Axioms
// =============================================================================

/// @brief \f$A A^* = A^* A\f$: unitarily diagonalizable, with an orthogonal eigenbasis.
///
/// The common ancestor of the self-adjoint, skew-adjoint and unitary families.
template <class Op, class X = Vector, class Y = Vector>
concept NormalOperator =
    LinearOperator<Op, X, Y> && Asserts<Op, property::normal>;

/// @brief \f$A = A^*\f$: symmetric over \f$\mathbb{R}\f$, Hermitian over \f$\mathbb{C}\f$.
///
/// Guarantees a real spectrum and an orthogonal eigenbasis — the precondition
/// MINRES and Lanczos actually rely on.
template <class Op, class X = Vector, class Y = Vector>
concept SelfAdjointOperator =
    NormalOperator<Op, X, Y> && Asserts<Op, property::self_adjoint>;

/// @brief \f$\langle x, A x \rangle \geq 0\f$: real non-negative spectrum, possibly singular.
///
/// Where graph Laplacians and Gram matrices belong. Distinguishing this from
/// `SPDOperator` matters: a Laplacian is not positive *definite*, and CG on one
/// stalls in its null space.
template <class Op, class X = Vector, class Y = Vector>
concept PSDOperator =
    SelfAdjointOperator<Op, X, Y> && Asserts<Op, property::psd>;

/// @brief \f$\langle x, A x \rangle > 0\f$ for \f$x \neq 0\f$: invertible, admits a Cholesky factor.
template <class Op, class X = Vector, class Y = Vector>
concept SPDOperator = PSDOperator<Op, X, Y> && Asserts<Op, property::spd>;

/// @brief \f$A = -A^*\f$: purely imaginary spectrum, \f$\langle x, A x \rangle = 0\f$ identically.
template <class Op, class X = Vector, class Y = Vector>
concept SkewAdjointOperator =
    NormalOperator<Op, X, Y> && Asserts<Op, property::skew_adjoint>;

/// @brief \f$A^* A = I\f$: an isometry, preserving inner products and hence norms.
template <class Op, class X = Vector, class Y = Vector>
concept UnitaryOperator =
    NormalOperator<Op, X, Y> && Asserts<Op, property::unitary>;

/// @brief \f$P = P^* = P^2\f$: the orthogonal projector onto its range.
template <class Op, class X = Vector, class Y = Vector>
concept ProjectionOperator =
    SelfAdjointOperator<Op, X, Y> && Asserts<Op, property::projection>;

// =============================================================================
// 3. Nonlinear maps
// =============================================================================

/// @brief Map \f$F: X \to Y\f$ between vector spaces carrying no linearity claim.
///
/// Deliberately not a weakening of `LinearOperator:` a nonlinear map admits
/// none of the axioms above, and superposition must not be assumed of it.
template <class Op, class X = Vector, class Y = Vector>
concept NonlinearOperator =
    VectorSpace<X> && MutableVectorSpace<Y> && requires(const Op &F, const X &x, Y &y) {
    { F.rows() } -> std::convertible_to<idx>;
    { F.cols() } -> std::convertible_to<idx>;
    { F.apply(x, y) };
};

/// @brief Linear operator that can materialize itself as explicit sparse storage.
template <class Op>
concept SparseConvertible = LinearOperator<Op> && requires(const Op &A) {
    { A.to_sparse() };
};

} // namespace num
