/// @file algebra/properties.hpp
/// @brief The property hierarchy: mathematical properties asserted about an object,
///        each bound to the runtime test that samples it.
///
/// Two different kinds of claim appear in numerical code and must not be conflated:
///
///   - **Structure** is *checked by the compiler*. That an object exposes an
///     `apply` taking a vector of its own scalar field is a syntactic fact, and a
///     concept can decide it.
///   - **Axioms** are *asserted by the caller*. That an operator is self-adjoint,
///     or positive definite, cannot be decided from a type. It is a promise, and
///     the most a library can do is sample it at runtime and reject violations.
///
/// This header carries the second kind. Each property is a type in an inheritance
/// hierarchy, so implication is structural. `spd` derives from `psd` derives from
/// `self_adjoint`, and a type tagged `spd` satisfies every weaker property without
/// restating them. Each property also owns a `verify`, so a property added without
/// a runtime test does not compile.
#pragma once

#include "algebra/debug.hpp"
#include "core/debug.hpp"
#include "core/types.hpp"
#include <concepts>
#include <source_location>
#include <string_view>
#include <type_traits>
#include <utility>

namespace num::property {

/// @brief \f$A(\alpha x + \beta y) = \alpha A x + \beta A y\f$. The root of the lattice.
struct linear {
    static constexpr std::string_view name = "linear";

    template <class Op, class V>
    static void verify(const Op &A, std::source_location loc) {
        debug::verify_linearity_sample<Op, V>(A, A.cols(), loc);
    }
};

/// @brief \f$A A^* = A^* A\f$. Normal operators are exactly those unitarily diagonalizable.
///
/// Normality has no cheap probe that does not require the adjoint, so it verifies
/// its base only; it exists to give the self-adjoint, skew-adjoint and unitary
/// families their common ancestor rather than leaving them siblings of `linear`.
struct normal : linear {
    static constexpr std::string_view name = "normal";

    template <class Op, class V>
    static void verify(const Op &A, std::source_location loc) {
        linear::verify<Op, V>(A, loc);
    }
};

/// @brief \f$A = A^*\f$ (symmetric over \f$\mathbb{R}\f$, Hermitian over \f$\mathbb{C}\f$).
struct self_adjoint : normal {
    static constexpr std::string_view name = "self_adjoint";

    template <class Op, class V>
    static void verify(const Op &A, std::source_location loc) {
        normal::verify<Op, V>(A, loc);
        debug::verify_symmetry_sample<Op, V>(A, A.cols(), loc);
    }
};

/// @brief \f$\langle x, A x \rangle \geq 0\f$. Admits a null space; Laplacians and Gram matrices
/// live here.
struct psd : self_adjoint {
    static constexpr std::string_view name = "psd";

    template <class Op, class V>
    static void verify(const Op &A, std::source_location loc) {
        self_adjoint::verify<Op, V>(A, loc);
        debug::verify_psd_sample<Op, V>(A, A.cols(), loc);
    }
};

/// @brief \f$\langle x, A x \rangle > 0\f$ for all \f$x \neq 0\f$. Invertible; admits a Cholesky
/// factor.
struct spd : psd {
    static constexpr std::string_view name = "spd";

    template <class Op, class V>
    static void verify(const Op &A, std::source_location loc) {
        self_adjoint::verify<Op, V>(A, loc);
        debug::verify_spd_sample<Op, V>(A, A.cols(), loc);
    }
};

/// @brief \f$A = -A^*\f$. The quadratic form vanishes identically; the spectrum is imaginary.
struct skew_adjoint : normal {
    static constexpr std::string_view name = "skew_adjoint";

    template <class Op, class V>
    static void verify(const Op &A, std::source_location loc) {
        normal::verify<Op, V>(A, loc);
        debug::verify_skew_symmetry_sample<Op, V>(A, A.cols(), loc);
    }
};

/// @brief \f$A^* A = I\f$: an isometry, preserving the inner product and hence the norm.
struct unitary : normal {
    static constexpr std::string_view name = "unitary";

    template <class Op, class V>
    static void verify(const Op &A, std::source_location loc) {
        normal::verify<Op, V>(A, loc);
        debug::verify_orthogonal_sample<Op, V>(A, A.cols(), loc);
    }
};

/// @brief \f$P^2 = P\f$ with \f$P = P^*\f$: the orthogonal projector onto its range.
struct projection : self_adjoint {
    static constexpr std::string_view name = "projection";

    template <class Op, class V>
    static void verify(const Op &A, std::source_location loc) {
        self_adjoint::verify<Op, V>(A, loc);
        debug::verify_projection_sample<Op, V>(A, A.cols(), loc);
    }
};

} // namespace num::property

namespace num {

/// @brief True when Obj carries a property tag at least as strong as P.
///
/// Implication comes from the lattice: a type declaring `using properties = property::spd`
/// satisfies `Asserts<Op, property::self_adjoint>` with no further declarations.
template <class Obj, class Ax>
concept Asserts = requires {
    typename std::remove_cvref_t<Obj>::properties;
}
&&std::derived_from<typename std::remove_cvref_t<Obj>::properties, Ax>;

/// @brief Run the sampled runtime test bound to property P, including every property it implies.
///
/// Cost is governed by the active diagnostic preset: full sampling under
/// `preset::strict`, nothing under `preset::production`.
template <class Ax, class V, class Obj>
inline void verify_property(const Obj &object,
                            std::source_location loc = std::source_location::current()) {
    Ax::template verify<Obj, V>(object, loc);
}

} // namespace num
