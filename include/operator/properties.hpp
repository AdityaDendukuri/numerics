/// @file operator/properties.hpp
/// @brief Attaching asserted mathematical properties to linear operators.
///
/// An axiom such as self-adjointness cannot be read off a type, so it enters the
/// type system as a claim: `assume_spd(A)` returns an operator that *says* it is
/// SPD. Two things make that claim trustworthy rather than decorative.
///
/// First, the claim is recorded as a position in the `num::law` hierarchy, so implication
/// is automatic — an operator asserted SPD satisfies every weaker law without restating
/// them, and `num::claims` is the only concept that has to read it.
///
/// Second, attaching the claim runs the sampled runtime test bound to that law, and to
/// every law it implies. `assume_spd` on a non-symmetric operator is rejected by the
/// inherited self-adjointness probe before definiteness is ever considered.
///
/// Whether those probes exist at all is a compile-time decision. `NUMERICS_DIAGNOSTICS`
/// defaults to 1 under `NDEBUG`, which leaves the shape checks and discards the sampling,
/// so an optimized build attaches the claim and runs no probe. Build with
/// `-DNUMERICS_DIAGNOSTICS=2` to keep sampling in an optimized build. See `core/debug.hpp`.
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

/// @brief A linear operator carrying an asserted axiom.
///
/// Replaces what was one hand-written wrapper class per property. The axiom is a
/// template parameter, so the hierarchy supplies the implications and there are no
/// per-property tag typedefs to keep in sync.
template <class Op, class Ax>
requires linear_operator<Op>
class structured_op final {
  public:
    /// @brief The law this operator claims. Implication comes from the hierarchy, so this
    /// one entry also supplies every weaker law.
    using math_laws = math::type_list<Ax>;
    using domain_type = vec;
    using codomain_type = vec;

    explicit structured_op(Op op,
                          std::source_location location = std::source_location::current())
        : op_(std::move(op)),
          provenance_{math::evidence_origin::assumed, location, "sampled assertion"} {}

    template <class X, class Y>
    void apply(const X &x, Y &y) const {
        op_.apply(x, y);
    }

    template <class Y, class X>
    void apply_adjoint(const Y &y, X &x) const requires adjointable_linear_operator<Op> {
        op_.apply_adjoint(y, x);
    }

    [[nodiscard]] idx rows() const noexcept { return op_.rows(); }
    [[nodiscard]] idx cols() const noexcept { return op_.cols(); }

    /// @brief The underlying operator, stripped of the assertion.
    [[nodiscard]] const Op &base() const noexcept { return op_; }
    [[nodiscard]] const math::evidence_provenance &provenance() const noexcept {
        return provenance_;
    }

  private:
    Op op_;
    math::evidence_provenance provenance_;
};

/// @brief Attach axiom Ax to an operator after sampling it and every axiom it implies.
template <class Ax, class V = vec, class Op>
requires linear_operator<Op>
[[nodiscard]] inline structured_op<Op, Ax>
assume(Op op, std::source_location loc = std::source_location::current()) {
    verify_property<Ax, V>(op, loc);
    return structured_op<Op, Ax>(std::move(op), loc);
}

// -----------------------------------------------------------------------------
// Named wrappers
// -----------------------------------------------------------------------------

/// @brief Operator asserted normal: \f$A A^* = A^* A\f$.
template <class Op>
using normal_op = structured_op<Op, law::normal>;

/// @brief Operator asserted self-adjoint: \f$A = A^*\f$.
template <class Op>
using symmetric_op = structured_op<Op, law::self_adjoint>;

/// @brief Operator asserted positive semi-definite: \f$\langle x, A x \rangle \geq 0\f$.
template <class Op>
using psd_op = structured_op<Op, law::psd>;

/// @brief Operator asserted positive definite: \f$\langle x, A x \rangle > 0\f$.
template <class Op>
using spd_op = structured_op<Op, law::spd>;

/// @brief Operator asserted unitary / orthogonal: \f$A^* A = I\f$.
template <class Op>
using orthogonal_op = structured_op<Op, law::unitary>;

/// @brief Operator asserted an orthogonal projector: \f$P = P^* = P^2\f$.
template <class Op>
using projection_op = structured_op<Op, law::projection>;

/// @brief Operator asserted skew-adjoint: \f$A = -A^*\f$.
template <class Op>
using skew_symmetric_op = structured_op<Op, law::skew_adjoint>;

// -----------------------------------------------------------------------------
// Assertion entry points
// -----------------------------------------------------------------------------

/// @brief Assert that `apply_adjoint` really is the adjoint of `apply`.
///
/// Verified by \f$\langle Ax, y \rangle = \langle x, A^* y \rangle\f$ over random
/// probes. A transposed-but-unconjugated adjoint passes on real data and fails
/// here on complex data, which is exactly where it would otherwise corrupt LSQR
/// and the Arnoldi recurrence.
template <class Op, class V = vec>
requires adjointable_linear_operator<Op>
[[nodiscard]] inline structured_op<Op, law::linear_map>
assume_adjointable(Op op, std::source_location loc = std::source_location::current()) {
    debug::verify_adjoint_sample<Op, V>(op, op.rows(), op.cols(), loc);
    return structured_op<Op, law::linear_map>(std::move(op), loc);
}

/// @brief Assert \f$A A^* = A^* A\f$.
template <class Op>
requires linear_operator<Op>
[[nodiscard]] inline normal_op<Op> assume_normal(Op op,
                                                std::source_location loc = std::source_location::current()) {
    return operators::assume<law::normal>(std::move(op), loc);
}

/// @brief Assert \f$A = A^*\f$; also samples linearity.
template <class Op>
requires linear_operator<Op>
[[nodiscard]] inline symmetric_op<Op>
assume_symmetric(Op op, std::source_location loc = std::source_location::current()) {
    return operators::assume<law::self_adjoint>(std::move(op), loc);
}

/// @brief Assert \f$\langle x, A x \rangle \geq 0\f$; also samples self-adjointness.
template <class Op>
requires linear_operator<Op>
[[nodiscard]] inline psd_op<Op> assume_psd(Op op,
                                          std::source_location loc = std::source_location::current()) {
    return operators::assume<law::psd>(std::move(op), loc);
}

/// @brief Assert \f$\langle x, A x \rangle > 0\f$; also samples self-adjointness and definiteness.
template <class Op>
requires linear_operator<Op>
[[nodiscard]] inline spd_op<Op> assume_spd(Op op,
                                          std::source_location loc = std::source_location::current()) {
    return operators::assume<law::spd>(std::move(op), loc);
}

/// @brief Assert \f$A^* A = I\f$.
template <class Op>
requires linear_operator<Op>
[[nodiscard]] inline orthogonal_op<Op>
assume_orthogonal(Op op, std::source_location loc = std::source_location::current()) {
    return operators::assume<law::unitary>(std::move(op), loc);
}

/// @brief Assert \f$P = P^* = P^2\f$.
template <class Op>
requires linear_operator<Op>
[[nodiscard]] inline projection_op<Op>
assume_projection(Op op, std::source_location loc = std::source_location::current()) {
    return operators::assume<law::projection>(std::move(op), loc);
}

/// @brief Assert \f$A = -A^*\f$.
template <class Op>
requires linear_operator<Op>
[[nodiscard]] inline skew_symmetric_op<Op>
assume_skew_symmetric(Op op, std::source_location loc = std::source_location::current()) {
    return operators::assume<law::skew_adjoint>(std::move(op), loc);
}

} // namespace num::operators

// No `claims_of` specialization: `structured_op` declares `math_laws` as a member, which
// takes precedence, and states the actual law rather than a weakened stand-in for it.
