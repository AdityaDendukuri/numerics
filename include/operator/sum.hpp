/// @file operator/sum.hpp
/// @brief The sum of two operators, carrying whatever law both operands carry.
///
/// \f$(A + B)x = Ax + Bx\f$, evaluated without forming a matrix. The point of the type is
/// not the arithmetic, which is two `apply` calls; it is that the sum keeps a law the
/// caller already established.
///
/// The rule is a theorem, and it is exactly the meet of the two operands' laws. If \f$A\f$
/// and \f$B\f$ are both positive definite then so is \f$A + B\f$, because
/// \f$\langle x, (A+B)x \rangle = \langle x, Ax \rangle + \langle x, Bx \rangle > 0\f$;
/// the same argument gives semidefiniteness, and self-adjointness follows from linearity
/// of the adjoint. Where the operands disagree the sum keeps only what both have: an SPD
/// operator plus a unitary one is normal and nothing stronger.
///
/// Without this the sum would claim nothing, and a caller wanting to solve with it would
/// have to re-assert definiteness -- paying an O(n^2) probe to re-establish a fact already
/// in hand, or asserting it unverified.
#pragma once

#include "core/math/concepts.hpp"
#include "core/math/models.hpp"
#include "core/math/operations.hpp"
#include <utility>

namespace num::operators {

/// @brief Non-owning \f$A + B\f$ for two operators over the same spaces.
template <class Lhs, class Rhs>
requires math::linear_operator<Lhs> && math::linear_operator<Rhs> &&
    std::same_as<math::domain_t<Lhs>, math::domain_t<Rhs>> &&
    std::same_as<math::codomain_t<Lhs>, math::codomain_t<Rhs>>
class sum_op final {
  public:
    using domain_type = math::domain_t<Lhs>;
    using codomain_type = math::codomain_t<Lhs>;

    sum_op(const Lhs &lhs, const Rhs &rhs) : lhs_(&lhs), rhs_(&rhs) {}

    void apply(const domain_type &x, codomain_type &y) const {
        math::apply(*lhs_, x, y);
        codomain_type t = math::zero_like(y);
        math::apply(*rhs_, x, t);
        math::axpy(scalar_t<codomain_type>(1), t, y);
    }

    [[nodiscard]] auto rows() const { return lhs_->rows(); }
    [[nodiscard]] auto cols() const { return lhs_->cols(); }
    [[nodiscard]] const Lhs &left() const noexcept { return *lhs_; }
    [[nodiscard]] const Rhs &right() const noexcept { return *rhs_; }

  private:
    const Lhs *lhs_;
    const Rhs *rhs_;
};

/// @brief Form \f$A + B\f$. Neither operand is copied, so both must outlive the result.
template <class Lhs, class Rhs>
[[nodiscard]] auto sum(const Lhs &lhs, const Rhs &rhs) {
    return sum_op<Lhs, Rhs>(lhs, rhs);
}

// Binding to a temporary would leave the sum holding a dangling operand at the end of the
// full expression, so rvalues are rejected rather than silently accepted.
template <class Lhs, class Rhs>
requires(!std::is_lvalue_reference_v<Lhs> || !std::is_lvalue_reference_v<Rhs>)
auto sum(Lhs &&, Rhs &&) = delete;

} // namespace num::operators

namespace num::math {

/// The sum satisfies whatever both operands satisfy: the meet of their laws.
template <class Lhs, class Rhs>
struct claims_of<operators::sum_op<Lhs, Rhs>> {
    using type = type_list<law::meet_t<law::strongest_law_t<Lhs>, law::strongest_law_t<Rhs>>>;
};

} // namespace num::math
