/// @file models.hpp
/// @brief The law hierarchy: every mathematical claim `numerics` can make, in one place.
///
/// A law is a statement about a type that the compiler cannot decide. Whether an operator
/// has an `apply` taking a vector is structure, and a `requires` clause settles it.
/// Whether that operator is self-adjoint is a law. No type can establish it. The caller
/// asserts it, and a runtime probe samples it.
///
/// Every law lives in the single hierarchy below. The laws are partially ordered by
/// implication, and C++ inheritance encodes that order, so a type claiming `law::spd`
/// satisfies every weaker law without restating it. `claims<T, L>` is the only concept
/// that reads the order.
///
/// The order is not a lattice. `law::spd` and `law::unitary` have a greatest lower bound,
/// `law::normal`, but no least upper bound: no law implies both. Only the meets exist, and
/// only those are used. Asking whether a type satisfies a law is a walk downward.
///
/// Two invariants are enforced by the structure rather than by convention:
///
///   - **Every law names its immediate base.** `verify_law` walks that chain, so a law's
///     probe cannot skip an intermediate law. Inserting a law between two existing ones
///     causes every stronger law to verify it, with no edit to any verifier.
///   - **Every law has a runtime test.** `law_verifier` is declared here and left
///     undefined. A law with no specialization is a compile error at the point of use
///     rather than a silently unchecked claim.
///
/// This header includes nothing beyond the standard library and defines no verifiers.
/// Verifiers need the sampling machinery, which sits above the containers. The hierarchy
/// stays at the bottom of the library so that any tier can name a law.
#pragma once

#include <complex>
#include <concepts>
#include <string_view>
#include <type_traits>

namespace num::math {

template <class... Ts>
struct type_list {};

} // namespace num::math

namespace num::law {

/// @brief Base of the roots of the hierarchy. `base = void` terminates the verification walk.
struct root {
    using base = void;
};

// -----------------------------------------------------------------------------
// Scalars
// -----------------------------------------------------------------------------

struct semiring : root {
    using base = void;
    static constexpr std::string_view name = "semiring";
};
struct ring : semiring {
    using base = semiring;
    static constexpr std::string_view name = "ring";
};
struct field : ring {
    using base = ring;
    static constexpr std::string_view name = "field";
};

// -----------------------------------------------------------------------------
// Spaces. Each is the one above it equipped with one further operation.
// -----------------------------------------------------------------------------

struct additive_group : root {
    using base = void;
    static constexpr std::string_view name = "additive_group";
};
/// @brief scalar action compatible with the group: \f$a(u+v) = au + av\f$.
struct vector_space : additive_group {
    using base = additive_group;
    static constexpr std::string_view name = "vector_space";
};
/// @brief Carries \f$\|\cdot\|\f$ with homogeneity and the triangle inequality.
struct normed_space : vector_space {
    using base = vector_space;
    static constexpr std::string_view name = "normed_space";
};
/// @brief Carries \f$\langle\cdot,\cdot\rangle\f$, sesquilinear and positive definite.
struct inner_product_space : normed_space {
    using base = normed_space;
    static constexpr std::string_view name = "inner_product_space";
};
/// @brief The norm is the one induced by the inner product: \f$\|x\|^2 = \langle x,x
/// \rangle\f$.
struct hilbert_space : inner_product_space {
    using base = inner_product_space;
    static constexpr std::string_view name = "hilbert_space";
};

/// @brief A subspace closed under the ambient space's operations.
struct linear_subspace : root {
    using base = void;
    static constexpr std::string_view name = "linear_subspace";
};

// -----------------------------------------------------------------------------
// Maps.
// -----------------------------------------------------------------------------

/// @brief \f$A(\alpha x + \beta y) = \alpha Ax + \beta Ay\f$. Root of the operator laws.
struct linear_map : root {
    using base = void;
    static constexpr std::string_view name = "linear_map";
};

/// @brief A linear map of a space into itself, so `rows() == cols()`.
struct endomorphism : linear_map {
    using base = linear_map;
    static constexpr std::string_view name = "endomorphism";
};

/// @brief \f$AA^* = A^*A\f$. These are exactly the unitarily diagonalizable operators.
///
/// Normality has no cheap probe that does not require the adjoint, so its own verifier
/// does nothing; it exists to give the self-adjoint, skew-adjoint and unitary families a
/// common ancestor instead of leaving them siblings of `linear_map`.
struct normal : endomorphism {
    using base = endomorphism;
    static constexpr std::string_view name = "normal";
};

/// @brief \f$A = A^*\f$: symmetric over \f$\mathbb{R}\f$, Hermitian over \f$\mathbb{C}\f$.
struct self_adjoint : normal {
    using base = normal;
    static constexpr std::string_view name = "self_adjoint";
};

/// @brief \f$\langle x, Ax \rangle \geq 0\f$. Admits a null space; Laplacians live here.
struct psd : self_adjoint {
    using base = self_adjoint;
    static constexpr std::string_view name = "psd";
};

/// @brief \f$\langle x, Ax \rangle > 0\f$ for \f$x \neq 0\f$: invertible, Cholesky-factorable.
struct spd : psd {
    using base = psd;
    static constexpr std::string_view name = "spd";
};

/// @brief \f$P = P^* = P^2\f$: the orthogonal projector onto its range.
///
/// Derives from `psd` rather than from `self_adjoint`. A projector is always positive
/// semidefinite, since \f$\langle x, Px \rangle = \langle x, P^2 x \rangle = \langle
/// Px, Px \rangle = \|Px\|^2 \geq 0\f$. Any routine accepting a PSD operator accepts a
/// projector.
struct projection : psd {
    using base = psd;
    static constexpr std::string_view name = "projection";
};

/// @brief \f$A = -A^*\f$: the quadratic form vanishes, the spectrum is imaginary.
struct skew_adjoint : normal {
    using base = normal;
    static constexpr std::string_view name = "skew_adjoint";
};

/// @brief \f$A^*A = I\f$: an isometry, preserving the inner product and hence the norm.
struct unitary : normal {
    using base = normal;
    static constexpr std::string_view name = "unitary";
};

// -----------------------------------------------------------------------------
// Restrictions to a subspace.
// -----------------------------------------------------------------------------
//
// An operator that is indefinite on the whole space is often definite on the subspace an
// algorithm iterates in. A graph Laplacian restricted to the zero-sum subspace is the
// common case. These laws state that without claiming it globally.

/// @brief The restriction to `Subspace` is self-adjoint and maps the subspace into itself.
template <class Subspace>
struct self_adjoint_on : endomorphism {
    using base = endomorphism;
    static constexpr std::string_view name = "self_adjoint_on";
};

/// @brief The restriction to `Subspace` is positive semidefinite.
template <class Subspace>
struct psd_on : self_adjoint_on<Subspace> {
    using base = self_adjoint_on<Subspace>;
    static constexpr std::string_view name = "psd_on";
};

/// @brief The restriction to `Subspace` is positive definite.
template <class Subspace>
struct spd_on : psd_on<Subspace> {
    using base = psd_on<Subspace>;
    static constexpr std::string_view name = "spd_on";
};

} // namespace num::law

namespace num::law {

// -----------------------------------------------------------------------------
// Deriving laws
// -----------------------------------------------------------------------------
//
// A law attached to an operand should survive an operation that provably preserves it.
// Without that, `A + B` for two SPD operators claims nothing, and the caller has to
// re-assert -- paying an O(n^2) sample, or asserting something unverified. The rules
// below are theorems, so the claim carries at compile time and costs nothing.
//
// Only *unconditional* theorems belong here. A congruence P^T A P preserves definiteness
// only when P has full column rank, and a product of self-adjoint operators is
// self-adjoint only when they commute. Neither side condition is checkable, so encoding
// them would be `assume` wearing a derivation's clothes. Those stay explicit.

/// @brief The strongest law implied by both `A` and `B`: their greatest lower bound.
///
/// The laws are partially ordered by implication and every pair has a meet, which is what
/// makes this total. It is exactly the rule for a sum: `A + B` satisfies whatever both
/// operands satisfy, and no more. `spd` with `spd` gives `spd`; `spd` with `self_adjoint`
/// gives `self_adjoint`; `spd` with `unitary` gives `normal`.
template <class A, class B>
struct meet {
    using type =
        std::conditional_t<std::derived_from<B, A>, A, typename meet<typename A::base, B>::type>;
};

template <class B>
struct meet<void, B> {
    using type = void;
};

template <class A, class B>
using meet_t = typename meet<A, B>::type;

/// @brief The law `L` restricted to `Subspace`, when one exists.
///
/// Projecting an operator onto a subspace does not preserve a global law: \f$P A\f$ is
/// not self-adjoint even when \f$A\f$ is. It preserves the law *on the subspace*, since
/// \f$PAx = PAPx\f$ for every \f$x \in S\f$, and \f$PAP\f$ is self-adjoint whenever
/// \f$A\f$ is. That is precisely what `self_adjoint_on` and its refinements assert.
template <class L, class Subspace>
struct restricted_to {
    using type = linear_map;
};
template <class S> struct restricted_to<spd, S>          { using type = spd_on<S>; };
template <class S> struct restricted_to<psd, S>          { using type = psd_on<S>; };
template <class S> struct restricted_to<projection, S>   { using type = psd_on<S>; };
template <class S> struct restricted_to<self_adjoint, S> { using type = self_adjoint_on<S>; };

template <class L, class Subspace>
using restricted_to_t = typename restricted_to<L, Subspace>::type;

/// @brief The most derived law in a claim list.
///
/// A type usually claims one law, and implication supplies the rest. Where it claims
/// several, the strongest is the one the others follow from. An empty list means the type
/// claimed nothing, which for a linear operator still means `linear_map`.
template <class List>
struct strongest;

template <>
struct strongest<math::type_list<>> {
    using type = linear_map;
};

template <class L>
struct strongest<math::type_list<L>> {
    using type = L;
};

template <class L, class... Rest>
struct strongest<math::type_list<L, Rest...>> {
  private:
    using rest = typename strongest<math::type_list<Rest...>>::type;

  public:
    using type = std::conditional_t<std::derived_from<L, rest>, L, rest>;
};


} // namespace num::law


namespace num::math {

/// @brief Alias for `num::law`, which was previously nested here as `num::math::law`.
///
/// The hierarchy is public vocabulary and now sits at `num::law`. Code inside `num::math`
/// names laws unqualified, so both spellings resolve to the one lattice.
namespace law = ::num::law;

/// @brief The runtime probe bound to a law.
///
/// Declared, never defined. Specializing it is how a law acquires its test, and the
/// absence of a specialization is a compile error the first time anything tries to verify
/// that law, so a law cannot be added without one. The specializations live in
/// `algebra/properties.hpp`, where the sampling machinery is visible.
///
/// A specialization tests only its own law. `verify_law` walks the weaker laws, which is
/// why an intermediate law cannot be skipped.
template <class Law>
struct law_verifier;

/// @brief Any law in the hierarchy above.
template <class L>
concept law_tag = requires {
    typename L::base;
    { L::name } -> std::convertible_to<std::string_view>;
};

} // namespace num::math

namespace num::math {

/// @brief The laws a type claims, for types whose definition you do not control.
///
/// Specialize this for foreign types (`std::vector`, `std::complex`, a third-party
/// matrix). A type you do own should instead declare a member
/// `using math_laws = num::math::type_list<...>;`, which takes precedence.
///
/// It lives in `num::math` rather than `num` because a specialization must be written in
/// the namespace of the primary template. The concept that reads it, `num::claims`, is
/// public vocabulary and stays in `num`.
template <class T>
struct claims_of {
    using type = type_list<>;
};

template <std::floating_point T>
struct claims_of<T> {
    using type = type_list<law::field>;
};

template <std::floating_point T>
struct claims_of<std::complex<T>> {
    using type = type_list<law::field>;
};

namespace detail {

template <class T, class = void>
struct declared_laws {
    using type = typename claims_of<std::remove_cvref_t<T>>::type;
};

template <class T>
struct declared_laws<T, std::void_t<typename std::remove_cvref_t<T>::math_laws>> {
    using type = typename std::remove_cvref_t<T>::math_laws;
};

template <class L, class... Ls>
consteval bool claims_in(type_list<Ls...>) {
    return (std::derived_from<Ls, L> || ...);
}

} // namespace detail

} // namespace num::math

namespace num::law {

/// @brief The most derived law a type claims, taken from its declaration.
///
/// Defined here rather than beside `strongest` because it needs `declared_laws`, which in
/// turn needs the claim customization point above.
template <class T>
using strongest_law_t =
    typename strongest<typename math::detail::declared_laws<T>::type>::type;

} // namespace num::law

namespace num {

using math::claims_of;

/// @brief True when `T` claims law `L`, or any law that implies it.
///
/// This is the only mechanism for reading a type's laws. It replaced three concepts over
/// three separate hierarchies: `Models`, which read `model_of<T>::laws`; `Carries`, which
/// read `T::math_propositions`; and `Asserts`, which read `T::properties`.
template <class T, class L>
concept claims = math::detail::claims_in<L>(typename math::detail::declared_laws<T>::type{});

} // namespace num
