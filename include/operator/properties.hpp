/// @file operator/properties.hpp
/// @brief Attaching asserted mathematical properties to linear operators.
///
/// An axiom such as self-adjointness cannot be read off a type, so it enters the
/// type system as a claim: `assume_spd(A)` returns an operator that *says* it is
/// SPD. Two things make that claim trustworthy rather than decorative.
///
/// First, the claim is recorded as a position in the `num::property` lattice, so
/// implication is automatic — an operator asserted SPD satisfies every weaker
/// axiom without restating them.
///
/// Second, attaching the claim runs the sampled runtime test bound to that axiom,
/// and to every axiom it implies. `assume_spd` on a non-symmetric operator is
/// rejected by the inherited self-adjointness probe before definiteness is ever
/// considered. Under `preset::production` the tests compile away entirely.
#pragma once

#include "algebra/debug.hpp"
#include "core/debug.hpp"
#include "algebra/properties.hpp"
#include "core/math/evidence.hpp"
#include "core/math/models.hpp"
#include "operator/concepts.hpp"
#include <type_traits>
#include <utility>

namespace num::operators {

namespace detail {

template <class Ax>
using mathematical_axioms = std::conditional_t<
    std::derived_from<Ax, property::spd>, math::type_list<axiom::positive_definite>,
    std::conditional_t<
        std::derived_from<Ax, property::psd>, math::type_list<axiom::positive_semidefinite>,
        std::conditional_t<std::derived_from<Ax, property::self_adjoint>,
                           math::type_list<axiom::self_adjoint>,
                           math::type_list<axiom::linear>>>>;

} // namespace detail

/// @brief A linear operator carrying an asserted axiom.
///
/// Replaces what was one hand-written wrapper class per property. The axiom is a
/// template parameter, so the lattice supplies the implications and there are no
/// per-property tag typedefs to keep in sync.
template <class Op, class Ax>
requires LinearOperator<Op>
class StructuredOp final {
  public:
    /// @brief Position of this operator in the property hierarchy.
    using properties = Ax;
    using domain_type = Vector;
    using codomain_type = Vector;
    using math_propositions = detail::mathematical_axioms<Ax>;

    explicit StructuredOp(Op op,
                          std::source_location location = std::source_location::current())
        : op_(std::move(op)),
          provenance_{math::evidence_origin::assumed, location, "legacy sampled assertion"} {}

    template <class X, class Y>
    void apply(const X &x, Y &y) const {
        op_.apply(x, y);
    }

    template <class Y, class X>
    void apply_adjoint(const Y &y, X &x) const requires AdjointableLinearOperator<Op> {
        op_.apply_adjoint(y, x);
    }

    [[nodiscard]] idx rows() const noexcept { return op_.rows(); }
    [[nodiscard]] idx cols() const noexcept { return op_.cols(); }

    /// @brief The underlying operator, stripped of the assertion.
    [[nodiscard]] const Op &base() const noexcept { return op_; }
    [[nodiscard]] const math::EvidenceProvenance &provenance() const noexcept {
        return provenance_;
    }

  private:
    Op op_;
    math::EvidenceProvenance provenance_;
};

/// @brief Attach axiom Ax to an operator after sampling it and every axiom it implies.
template <class Ax, class V = Vector, class Op>
requires LinearOperator<Op>
[[nodiscard]] inline StructuredOp<Op, Ax>
assume(Op op, std::source_location loc = std::source_location::current()) {
    verify_property<Ax, V>(op, loc);
    return StructuredOp<Op, Ax>(std::move(op), loc);
}

// -----------------------------------------------------------------------------
// Named wrappers
// -----------------------------------------------------------------------------

/// @brief Operator asserted normal: \f$A A^* = A^* A\f$.
template <class Op>
using NormalOp = StructuredOp<Op, property::normal>;

/// @brief Operator asserted self-adjoint: \f$A = A^*\f$.
template <class Op>
using SymmetricOp = StructuredOp<Op, property::self_adjoint>;

/// @brief Operator asserted positive semi-definite: \f$\langle x, A x \rangle \geq 0\f$.
template <class Op>
using PSDOp = StructuredOp<Op, property::psd>;

/// @brief Operator asserted positive definite: \f$\langle x, A x \rangle > 0\f$.
template <class Op>
using SPDOp = StructuredOp<Op, property::spd>;

/// @brief Operator asserted unitary / orthogonal: \f$A^* A = I\f$.
template <class Op>
using OrthogonalOp = StructuredOp<Op, property::unitary>;

/// @brief Operator asserted an orthogonal projector: \f$P = P^* = P^2\f$.
template <class Op>
using ProjectionOp = StructuredOp<Op, property::projection>;

/// @brief Operator asserted skew-adjoint: \f$A = -A^*\f$.
template <class Op>
using SkewSymmetricOp = StructuredOp<Op, property::skew_adjoint>;

// -----------------------------------------------------------------------------
// Assertion entry points
// -----------------------------------------------------------------------------

/// @brief Assert that `apply_adjoint` really is the adjoint of `apply`.
///
/// Verified by \f$\langle Ax, y \rangle = \langle x, A^* y \rangle\f$ over random
/// probes. A transposed-but-unconjugated adjoint passes on real data and fails
/// here on complex data, which is exactly where it would otherwise corrupt LSQR
/// and the Arnoldi recurrence.
template <class Op, class V = Vector>
requires AdjointableLinearOperator<Op>
[[nodiscard]] inline StructuredOp<Op, property::linear>
assume_adjointable(Op op, std::source_location loc = std::source_location::current()) {
    debug::verify_adjoint_sample<Op, V>(op, op.rows(), op.cols(), loc);
    return StructuredOp<Op, property::linear>(std::move(op), loc);
}

/// @brief Assert \f$A A^* = A^* A\f$.
template <class Op>
requires LinearOperator<Op>
[[nodiscard]] inline NormalOp<Op> assume_normal(Op op,
                                                std::source_location loc = std::source_location::current()) {
    return operators::assume<property::normal>(std::move(op), loc);
}

/// @brief Assert \f$A = A^*\f$; also samples linearity.
template <class Op>
requires LinearOperator<Op>
[[nodiscard]] inline SymmetricOp<Op>
assume_symmetric(Op op, std::source_location loc = std::source_location::current()) {
    return operators::assume<property::self_adjoint>(std::move(op), loc);
}

/// @brief Assert \f$\langle x, A x \rangle \geq 0\f$; also samples self-adjointness.
template <class Op>
requires LinearOperator<Op>
[[nodiscard]] inline PSDOp<Op> assume_psd(Op op,
                                          std::source_location loc = std::source_location::current()) {
    return operators::assume<property::psd>(std::move(op), loc);
}

/// @brief Assert \f$\langle x, A x \rangle > 0\f$; also samples self-adjointness and definiteness.
template <class Op>
requires LinearOperator<Op>
[[nodiscard]] inline SPDOp<Op> assume_spd(Op op,
                                          std::source_location loc = std::source_location::current()) {
    return operators::assume<property::spd>(std::move(op), loc);
}

/// @brief Assert \f$A^* A = I\f$.
template <class Op>
requires LinearOperator<Op>
[[nodiscard]] inline OrthogonalOp<Op>
assume_orthogonal(Op op, std::source_location loc = std::source_location::current()) {
    return operators::assume<property::unitary>(std::move(op), loc);
}

/// @brief Assert \f$P = P^* = P^2\f$.
template <class Op>
requires LinearOperator<Op>
[[nodiscard]] inline ProjectionOp<Op>
assume_projection(Op op, std::source_location loc = std::source_location::current()) {
    return operators::assume<property::projection>(std::move(op), loc);
}

/// @brief Assert \f$A = -A^*\f$.
template <class Op>
requires LinearOperator<Op>
[[nodiscard]] inline SkewSymmetricOp<Op>
assume_skew_symmetric(Op op, std::source_location loc = std::source_location::current()) {
    return operators::assume<property::skew_adjoint>(std::move(op), loc);
}

} // namespace num::operators

namespace num::math {

template <class Op, class Ax>
struct model_of<operators::StructuredOp<Op, Ax>> {
    using laws = type_list<law::linear_map>;
};

} // namespace num::math
