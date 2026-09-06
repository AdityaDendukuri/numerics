/// @file algebra/properties.hpp
/// @brief Binding each law in the hierarchy to the runtime test that samples it.
///
/// The hierarchy itself is in `core/math/models.hpp` and has no dependencies, so anything
/// in the library may name a law. The probes need random vectors and operator
/// application, which live above the containers, so they are attached here by specializing
/// `math::law_verifier`.
///
/// Two rules follow from the structure rather than from convention.
///
///   - A specialization tests **only its own law**. `verify_law` below walks the weaker
///     laws by reading `L::base`. The walk is not written by hand at each level, so a
///     probe cannot skip an intermediate law.
///   - `math::law_verifier` is declared undefined in the hierarchy header. A law with no
///     specialization here is a compile error the first time anything verifies it, so a
///     law cannot be added without a test.
///
/// Sampling `⟨x, Ax⟩ > 0` over a basis and some random directions refutes a false claim
/// cheaply and often. It never proves a true one. These probes reject; they do not
/// certify. The claim remains the caller's.
#pragma once

#include "algebra/debug.hpp"
#include "core/debug.hpp"
#include "core/math/models.hpp"
#include "core/types.hpp"
#include <concepts>
#include <source_location>
#include <type_traits>
#include <utility>

namespace num::math {

// -----------------------------------------------------------------------------
// scalar and space laws
// -----------------------------------------------------------------------------
//
// These are checked structurally rather than by sampling. The concept that admits a type
// into the space hierarchy already requires the defining operations to exist and to be
// callable with the right types, which is all a compiler can decide. The remaining
// content, such as associativity, distributivity and the triangle inequality, is a
// numerical property of a specific implementation. A probe for it would belong here and
// none exists yet, so these verifiers are empty rather than misleading.

#define NUM_STRUCTURAL_LAW(LAW)                                                            \
    template <>                                                                            \
    struct law_verifier<LAW> {                                                             \
        template <class Obj, class V>                                                      \
        static void run(const Obj &, std::source_location) noexcept {}                     \
    }

NUM_STRUCTURAL_LAW(law::semiring);
NUM_STRUCTURAL_LAW(law::ring);
NUM_STRUCTURAL_LAW(law::field);
NUM_STRUCTURAL_LAW(law::additive_group);
NUM_STRUCTURAL_LAW(law::vector_space);
NUM_STRUCTURAL_LAW(law::normed_space);
NUM_STRUCTURAL_LAW(law::inner_product_space);
NUM_STRUCTURAL_LAW(law::hilbert_space);
NUM_STRUCTURAL_LAW(law::linear_subspace);

#undef NUM_STRUCTURAL_LAW

// -----------------------------------------------------------------------------
// Operator laws
// -----------------------------------------------------------------------------

/// @brief \f$A(\alpha x + \beta y) = \alpha Ax + \beta Ay\f$, sampled on random pairs.
template <>
struct law_verifier<law::linear_map> {
    template <class Op, class V>
    static void run(const Op &A, std::source_location loc) {
        debug::verify_linearity_sample<Op, V>(A, A.cols(), loc);
    }
};

/// @brief Squareness. Decidable at runtime from the shape alone, so this one is exact.
template <>
struct law_verifier<law::endomorphism> {
    template <class Op, class V>
    static void run(const Op &A, std::source_location loc) {
        debug::check_dim(A.rows(), A.cols(), "operator claimed as an endomorphism is not square",
                         loc);
    }
};

/// @brief No probe: normality cannot be sampled without the adjoint.
///
/// The law is still useful. It makes self-adjoint, skew-adjoint and unitary siblings under
/// a common ancestor rather than three unrelated refinements of `linear_map`, so an
/// algorithm that needs only "unitarily diagonalizable" can say so.
template <>
struct law_verifier<law::normal> {
    template <class Op, class V>
    static void run(const Op &, std::source_location) noexcept {}
};

/// @brief \f$\langle Ax, y \rangle = \langle x, Ay \rangle\f$ over sampled pairs.
template <>
struct law_verifier<law::self_adjoint> {
    template <class Op, class V>
    static void run(const Op &A, std::source_location loc) {
        debug::verify_symmetry_sample<Op, V>(A, A.cols(), loc);
    }
};

/// @brief \f$\langle x, Ax \rangle \geq 0\f$.
template <>
struct law_verifier<law::psd> {
    template <class Op, class V>
    static void run(const Op &A, std::source_location loc) {
        debug::verify_psd_sample<Op, V>(A, A.cols(), loc);
    }
};

/// @brief \f$\langle x, Ax \rangle > 0\f$ for \f$x \neq 0\f$.
template <>
struct law_verifier<law::spd> {
    template <class Op, class V>
    static void run(const Op &A, std::source_location loc) {
        debug::verify_spd_sample<Op, V>(A, A.cols(), loc);
    }
};

/// @brief \f$P^2 = P\f$. Self-adjointness and semidefiniteness come from the base walk.
template <>
struct law_verifier<law::projection> {
    template <class Op, class V>
    static void run(const Op &P, std::source_location loc) {
        debug::verify_projection_sample<Op, V>(P, P.cols(), loc);
    }
};

/// @brief \f$\langle Ax, y \rangle = -\langle x, Ay \rangle\f$.
template <>
struct law_verifier<law::skew_adjoint> {
    template <class Op, class V>
    static void run(const Op &A, std::source_location loc) {
        debug::verify_skew_symmetry_sample<Op, V>(A, A.cols(), loc);
    }
};

/// @brief \f$\|Ax\| = \|x\|\f$ over sampled directions.
template <>
struct law_verifier<law::unitary> {
    template <class Op, class V>
    static void run(const Op &A, std::source_location loc) {
        debug::verify_orthogonal_sample<Op, V>(A, A.cols(), loc);
    }
};

// -----------------------------------------------------------------------------
// Subspace restrictions
// -----------------------------------------------------------------------------
//
// A claim about the restriction of an operator to a subspace cannot be sampled with
// vectors drawn from the ambient space. The probe would have to project first, and the
// projector belongs to the subspace type rather than to the operator. These are verified
// structurally. `math::assume` still enforces squareness through the base walk.

template <class S>
struct law_verifier<law::self_adjoint_on<S>> {
    template <class Op, class V>
    static void run(const Op &, std::source_location) noexcept {}
};
template <class S>
struct law_verifier<law::psd_on<S>> {
    template <class Op, class V>
    static void run(const Op &, std::source_location) noexcept {}
};
template <class S>
struct law_verifier<law::spd_on<S>> {
    template <class Op, class V>
    static void run(const Op &, std::source_location) noexcept {}
};

// -----------------------------------------------------------------------------
// The walk
// -----------------------------------------------------------------------------

/// @brief Verify `L` and every law it implies, weakest first.
///
/// The chain comes from `L::base`, so this is the only place the ordering is expressed.
/// Adding a law between two existing ones makes every stronger law verify it with no edit
/// to any verifier. This prevents the class of bug in which `spd` tested self-adjointness
/// and definiteness but silently skipped semidefiniteness.
///
/// @tparam L   The law to verify.
/// @tparam Obj The object claiming it.
/// @tparam V   vec type the probes should draw from.
template <class L, class Obj, class V>
inline void verify_law(const Obj &object,
                       std::source_location loc = std::source_location::current()) {
    if constexpr (!std::is_void_v<typename L::base>) {
        verify_law<typename L::base, Obj, V>(object, loc);
    }
    law_verifier<L>::template run<Obj, V>(object, loc);
}

} // namespace num::math

namespace num {

using math::verify_law;

/// @brief Run the sampled test bound to law `Ax`, and to every law it implies.
///
/// Cost is governed by the active diagnostic preset; see `core/debug.hpp`.
template <class Ax, class V, class Obj>
inline void verify_property(const Obj &object,
                            std::source_location loc = std::source_location::current()) {
    math::verify_law<Ax, Obj, V>(object, loc);
}

} // namespace num
