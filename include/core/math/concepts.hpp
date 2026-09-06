/// @file concepts.hpp
/// @brief The concept hierarchy: a scalar field, successively equipped with more structure.
///
/// Every concept here refines the one above it, adding one operation the compiler can check
/// or one law the caller asserts. A normed space is a vector space equipped with a norm;
/// an inner product space is a normed space equipped with an inner product; and so on.
///
/// No concept in this library asserts a mathematical property with a bare `requires`
/// block. It names the concept it refines and adds only what is new.
///
/// ```
/// field<T>                    + - * /, 0, 1
///  └ additive_group<V>         dimension, zero_like, copy
///     └ vector_space<V>        + scale, axpy                 ── law::vector_space
///        └ normed_space<V>     + norm                        ── law::normed_space
///           └ inner_product_space<V>  + inner                 ── law::inner_product_space
///              └ hilbert_space<V>    ‖x‖² = ⟨x,x⟩            ── law::hilbert_space
///
/// linear_map<Op>               domain and codomain are spaces ── law::linear_map
///  └ linear_operator<Op>       + apply, rows, cols
///     └ endomorphism<Op>      domain == codomain             ── law::endomorphism
///        └ normal_operator     AA* = A*A                      ── law::normal
///           ├ self_adjoint_operator  A = A*                    ── law::self_adjoint
///           │  └ psd_operator        ⟨x,Ax⟩ ≥ 0               ── law::psd
///           │     ├ spd_operator      ⟨x,Ax⟩ > 0              ── law::spd
///           │     └ projection_operator  P = P* = P²          ── law::projection
///           ├ skew_adjoint_operator  A = −A*                   ── law::skew_adjoint
///           └ unitary_operator      A*A = I                   ── law::unitary
/// ```
///
/// The `requires` clauses decide structure. Laws come from `claims<T, L>` and are never
/// inferred from syntax. A type therefore cannot become SPD by accident. It must declare
/// the law, and declaring it runs the probe bound to that law.
///
/// The operator concepts default their domain and codomain to the type's own associated
/// types. `linear_operator<Op>` and `linear_operator<Op, vec, vec>` are the same
/// concept with the spaces left implicit or stated.
///
/// This header depends on nothing but the standard library, so the whole hierarchy is
/// available to any tier, including code that never touches `num::vec`.
#pragma once

#include "core/math/associated.hpp"
#include "core/math/models.hpp"
#include "core/math/operations.hpp"
#include <concepts>
#include <type_traits>

namespace num::math {

// -----------------------------------------------------------------------------
// Scalars
// -----------------------------------------------------------------------------

/// @brief scalar field \f$\mathbb{K}\f$: closed under `+ - * /` with identities 0 and 1.
template <class T>
concept field = claims<T, law::field> && requires(T a, T b) {
    {a + b}->std::convertible_to<T>;
    {a - b}->std::convertible_to<T>;
    {a * b}->std::convertible_to<T>;
    {a / b}->std::convertible_to<T>;
    T{0};
    T{1};
};

// -----------------------------------------------------------------------------
// Spaces
// -----------------------------------------------------------------------------

/// @brief Abelian group under addition, over a scalar field: sized, copyable, zeroable.
///
/// The structural floor of the hierarchy. It does not require scalar multiplication, which
/// `vector_space` adds. A type that can be added and measured but not scaled still has a
/// level to stand on.
template <class V>
concept additive_group = field<scalar_t<V>> && std::copy_constructible<V> &&
    requires(const V &x) {
    {dimension(x)}->std::integral;
    {zero_like(x)}->std::same_as<V>;
};

/// @brief vec space over \f$\mathbb{K}\f$: an additive group with a compatible scalar action.
template <class V>
concept vector_space = additive_group<V> && claims<V, law::vector_space> &&
    requires(V &v, const V &x, scalar_t<V> a) {
    scale(a, v);
    axpy(a, x, v);
};

/// @brief vec space carrying a norm \f$\|\cdot\|\f$.
template <class V>
concept normed_space = vector_space<V> && claims<V, law::normed_space> && requires(const V &v) {
    norm(v);
};

/// @brief Normed space carrying an inner product \f$\langle\cdot,\cdot\rangle\f$.
template <class V>
concept inner_product_space =
    normed_space<V> && claims<V, law::inner_product_space> && requires(const V &x, const V &y) {
    {inner(x, y)}->std::convertible_to<scalar_t<V>>;
};

/// @brief Inner product space whose norm is the induced one: \f$\|x\|^2 = \langle x,x\rangle\f$.
///
/// The setting every Krylov method needs. CG, MINRES and GMRES all require that the norm
/// they minimize is the one the inner product induces.
template <class V>
concept hilbert_space = inner_product_space<V> && claims<V, law::hilbert_space>;

/// @brief vec space whose coordinates can be written individually.
template <class V>
concept mutable_vector_space = vector_space<V> && requires(V &v, scalar_t<V> a) {
    { v[std::size_t{0}] = a };
};

/// @brief vec space backed by one contiguous run of scalars.
///
/// A statement about storage rather than mathematics. It allows an algorithm to drop
/// through to `num::kernel`, which operates only on pointers.
template <class V>
concept contiguous_vector = vector_space<V> && requires(V & v, const V &cv) {
    {v.data()}->std::convertible_to<scalar_t<V> *>;
    {cv.data()}->std::convertible_to<const scalar_t<V> *>;
};

// -----------------------------------------------------------------------------
// Maps
// -----------------------------------------------------------------------------

/// @brief Linear map \f$A: X \to Y\f$ between vector spaces.
template <class Op, class X = domain_t<Op>, class Y = codomain_t<Op>>
concept linear_map = claims<Op, law::linear_map> && vector_space<X> && vector_space<Y>;

/// @brief Linear map that can be evaluated and knows its shape.
template <class Op, class X = domain_t<Op>, class Y = codomain_t<Op>>
concept linear_operator = linear_map<Op, X, Y> && requires(const Op &A, const X &x, Y &y) {
    // The CPO rather than `A.apply(x, y)`. It also resolves a `tag_invoke` overload and
    // unwraps evidence wrappers such as `certified_ref`, which carry a law about a matrix
    // without re-exposing its interface.
    apply(A, x, y);
    {A.rows()}->std::integral;
    {A.cols()}->std::integral;
};

/// @brief Linear operator exposing its adjoint \f$A^*\f$.
template <class Op, class X = domain_t<Op>, class Y = codomain_t<Op>>
concept adjointable_linear_operator =
    linear_operator<Op, X, Y> && requires(const Op &A, const Y &y, X &x) {
    A.apply_adjoint(y, x);
};

/// @brief Linear operator of a space into itself, so `rows() == cols()`.
template <class Op, class V = domain_t<Op>>
concept endomorphism =
    linear_operator<Op, V, V> && claims<Op, law::endomorphism> && std::same_as<domain_t<Op>, V>;

/// @brief An operator that maps `V` into itself, whether or not it claims `law::endomorphism`.
///
/// The structural half of `endomorphism`. It is kept separate so a solver signature can
/// require that an operator acts on a given space without also requiring the squareness
/// claim.
template <class Op, class V>
concept endomorphism_on =
    linear_operator<Op, V, V> && std::same_as<domain_t<Op>, V> && std::same_as<codomain_t<Op>, V>;

// -----------------------------------------------------------------------------
// Operator laws
// -----------------------------------------------------------------------------
//
// A subsumption chain, so overload resolution orders correctly: an SPD operator offered
// to a set containing CG, MINRES and GMRES selects CG rather than being ambiguous.

/// @brief \f$AA^* = A^*A\f$: unitarily diagonalizable, with an orthogonal eigenbasis.
template <class Op, class X = domain_t<Op>, class Y = codomain_t<Op>>
concept normal_operator = linear_operator<Op, X, Y> && claims<Op, law::normal>;

/// @brief \f$A = A^*\f$: real spectrum. The precondition for MINRES and Lanczos.
template <class Op, class X = domain_t<Op>, class Y = codomain_t<Op>>
concept self_adjoint_operator = normal_operator<Op, X, Y> && claims<Op, law::self_adjoint>;

/// @brief \f$\langle x, Ax \rangle \ge 0\f$: graph Laplacians and Gram matrices, possibly singular.
template <class Op, class X = domain_t<Op>, class Y = codomain_t<Op>>
concept psd_operator = self_adjoint_operator<Op, X, Y> && claims<Op, law::psd>;

/// @brief \f$\langle x, Ax \rangle > 0\f$: invertible, Cholesky-factorable. The precondition for CG.
template <class Op, class X = domain_t<Op>, class Y = codomain_t<Op>>
concept spd_operator = psd_operator<Op, X, Y> && claims<Op, law::spd>;

/// @brief \f$P = P^* = P^2\f$: the orthogonal projector onto its range.
///
/// Refines `psd_operator` rather than `self_adjoint_operator`. A projector is always positive
/// semidefinite, since \f$\langle x, Px \rangle = \langle Px, Px \rangle \ge 0\f$.
template <class Op, class X = domain_t<Op>, class Y = codomain_t<Op>>
concept projection_operator = psd_operator<Op, X, Y> && claims<Op, law::projection>;

/// @brief \f$A = -A^*\f$: imaginary spectrum. Conservative advection, Hamiltonian fields.
template <class Op, class X = domain_t<Op>, class Y = codomain_t<Op>>
concept skew_adjoint_operator = normal_operator<Op, X, Y> && claims<Op, law::skew_adjoint>;

/// @brief \f$A^*A = I\f$: an isometry. Givens rotations, Householder reflectors, the DFT.
template <class Op, class X = domain_t<Op>, class Y = codomain_t<Op>>
concept unitary_operator = normal_operator<Op, X, Y> && claims<Op, law::unitary>;

// -----------------------------------------------------------------------------
// Maps carrying no linearity claim
// -----------------------------------------------------------------------------

/// @brief \f$F: X \to Y\f$ between vector spaces, with no claim of linearity.
template <class Op, class X = domain_t<Op>, class Y = codomain_t<Op>>
concept nonlinear_operator =
    vector_space<X> && mutable_vector_space<Y> && requires(const Op &F, const X &x, Y &y) {
    apply(F, x, y);
    {F.rows()}->std::integral;
    {F.cols()}->std::integral;
};

} // namespace num::math

namespace num {

// The hierarchy is public vocabulary. It is defined in `num::math` because that is where the
// customization points and the associated-type traits live, and re-exported here so
// callers write `num::spd_operator` rather than reaching into the machinery namespace.
using math::additive_group;
using math::adjointable_linear_operator;
using math::contiguous_vector;
using math::endomorphism;
using math::endomorphism_on;
using math::field;
using math::hilbert_space;
using math::inner_product_space;
using math::linear_map;
using math::linear_operator;
using math::mutable_vector_space;
using math::nonlinear_operator;
using math::normal_operator;
using math::normed_space;
using math::projection_operator;
using math::psd_operator;
using math::self_adjoint_operator;
using math::skew_adjoint_operator;
using math::spd_operator;
using math::unitary_operator;
using math::vector_space;

} // namespace num
