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
/// Linearity is the foundational contract for matrix-free and explicit operators:
/// requires queryable dimensions `rows()` and `cols()` and an out-parameter evaluation method `apply(x, y)`.
///
/// ### Example
/// @code
/// auto op = num::operators::make_op([](const num::Vector& x, num::Vector& y) {
///     // compute y <- A * x
/// }, n, n);
/// static_assert(num::LinearOperator<decltype(op)>);
/// @endcode
///
/// @tparam Op Operator or matrix wrapper type.
/// @tparam X Domain vector space (defaults to `Vector`).
/// @tparam Y Codomain vector space (defaults to `Vector`).
template <class Op, class X = Vector, class Y = Vector>
concept LinearOperator =
    VectorSpace<X> && MutableVectorSpace<Y> && requires(const Op &A, const X &x, Y &y) {
    { A.rows() } -> std::convertible_to<idx>;
    { A.cols() } -> std::convertible_to<idx>;
    { A.apply(x, y) };
};

/// @brief Linear map exposing its Hermitian adjoint \f$A^*\f$, with \f$\langle A x, y \rangle = \langle x, A^* y \rangle\f$.
///
/// In addition to forward evaluation `apply(x, y)`, exposes `apply_adjoint(y, x)`.
///
/// @tparam Op Operator type.
/// @tparam X Domain vector space.
/// @tparam Y Codomain vector space.
template <class Op, class X = Vector, class Y = Vector>
concept AdjointableLinearOperator =
    LinearOperator<Op, X, Y> && requires(const Op &A, const Y &y, X &x) {
    { A.apply_adjoint(y, x) };
};

// =============================================================================
// 2. Axioms
// =============================================================================

/// @brief Normal operator \f$A A^* = A^* A\f$: unitarily diagonalizable with an orthogonal eigenbasis.
///
/// The algebraic common ancestor of self-adjoint, skew-adjoint, and unitary operator families.
///
/// @tparam Op Operator type.
/// @tparam X Domain vector space.
/// @tparam Y Codomain vector space.
template <class Op, class X = Vector, class Y = Vector>
concept NormalOperator =
    LinearOperator<Op, X, Y> && Asserts<Op, property::normal>;

/// @brief Self-adjoint operator \f$A = A^*\f$ (symmetric over \f$\mathbb{R}\f$, Hermitian over \f$\mathbb{C}\f$).
///
/// Guarantees a real spectrum \f$\sigma(A) \subset \mathbb{R}\f$ and an orthogonal eigenbasis.
/// This is the mathematical precondition required by **MINRES** (`num::minres`) and **Lanczos** (`num::lanczos`).
///
/// ### Tagging Evidence
/// Attach evidence via `num::operators::assume_symmetric(op)` or `num::assume<num::axiom::self_adjoint>(op)`.
///
/// @tparam Op Operator type.
/// @tparam X Domain vector space.
/// @tparam Y Codomain vector space.
template <class Op, class X = Vector, class Y = Vector>
concept SelfAdjointOperator =
    NormalOperator<Op, X, Y> && Asserts<Op, property::self_adjoint>;

/// @brief Positive semi-definite operator \f$\langle x, A x \rangle \ge 0\f$: real non-negative spectrum.
///
/// Characterizes graph Laplacians (\f$L = D - A\f$) and Gram matrices (\f$A^T A\f$).
/// May be singular with non-trivial nullspace (e.g. constant null vector \f$\mathbf{1}\f$).
///
/// @tparam Op Operator type.
/// @tparam X Domain vector space.
/// @tparam Y Codomain vector space.
template <class Op, class X = Vector, class Y = Vector>
concept PSDOperator =
    SelfAdjointOperator<Op, X, Y> && Asserts<Op, property::psd>;

/// @brief Symmetric Positive Definite (SPD) operator: \f$\langle x, A x \rangle > 0\f$ for all \f$x \ne 0\f$.
///
/// Strictly positive spectrum \f$\sigma(A) \subset (0, \infty)\f$. Strictly invertible, guaranteeing
/// stable Cholesky factorization and monotonic energy norm minimization in **Conjugate Gradient (`num::cg`)**.
///
/// ### Tagging Evidence
/// Attach evidence via `num::operators::assume_spd(op)` or instantiate certified operators like `BackwardEuler2D`.
///
/// @tparam Op Operator type.
/// @tparam X Domain vector space.
/// @tparam Y Codomain vector space.
template <class Op, class X = Vector, class Y = Vector>
concept SPDOperator = PSDOperator<Op, X, Y> && Asserts<Op, property::spd>;

/// @brief Skew-adjoint operator \f$A = -A^*\f$: purely imaginary spectrum \f$\sigma(A) \subset i\mathbb{R}\f$.
///
/// Represents conservative advection, Hamiltonian vector fields, and generators of unitary groups.
///
/// @tparam Op Operator type.
/// @tparam X Domain vector space.
/// @tparam Y Codomain vector space.
template <class Op, class X = Vector, class Y = Vector>
concept SkewAdjointOperator =
    NormalOperator<Op, X, Y> && Asserts<Op, property::skew_adjoint>;

/// @brief Unitary operator \f$A^* A = I\f$: isometric operator preserving inner products and norms.
///
/// Satisfies \f$\|A x\| = \|x\|\f$. Examples include Givens rotations, Householder reflectors, and discrete Fourier transforms (FFT).
///
/// @tparam Op Operator type.
/// @tparam X Domain vector space.
/// @tparam Y Codomain vector space.
template <class Op, class X = Vector, class Y = Vector>
concept UnitaryOperator =
    NormalOperator<Op, X, Y> && Asserts<Op, property::unitary>;

/// @brief Projection operator \f$P = P^* = P^2\f$: orthogonal projector onto its range.
///
/// Idempotent and self-adjoint. Eigenvalues are strictly contained in \f$\{0, 1\}\f$.
///
/// @tparam Op Operator type.
/// @tparam X Domain vector space.
/// @tparam Y Codomain vector space.
template <class Op, class X = Vector, class Y = Vector>
concept ProjectionOperator =
    SelfAdjointOperator<Op, X, Y> && Asserts<Op, property::projection>;

// =============================================================================
// 3. Nonlinear maps
// =============================================================================

/// @brief Map \f$F: X \to Y\f$ between vector spaces carrying no linearity claim.
///
/// @tparam Op Operator type.
/// @tparam X Domain vector space.
/// @tparam Y Codomain vector space.
template <class Op, class X = Vector, class Y = Vector>
concept NonlinearOperator =
    VectorSpace<X> && MutableVectorSpace<Y> && requires(const Op &F, const X &x, Y &y) {
    { F.rows() } -> std::convertible_to<idx>;
    { F.cols() } -> std::convertible_to<idx>;
    { F.apply(x, y) };
};

/// @brief Linear operator that can materialize itself as explicit sparse CSR storage.
///
/// @tparam Op Operator type.
template <class Op>
concept SparseConvertible = LinearOperator<Op> && requires(const Op &A) {
    { A.to_sparse() };
};

} // namespace num
