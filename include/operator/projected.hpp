/// @file projected.hpp
/// @brief Operator adapter that projects every output onto a linear subspace.
#pragma once

#include "core/math/associated.hpp"
#include "core/math/concepts.hpp"
#include "core/math/models.hpp"
#include "core/math/subspace.hpp"
#include <type_traits>
#include <utility>

namespace num::operators {

/// Non-owning representation of P_S A, where P_S is projection onto S.
template <class Op, class Subspace>
requires math::linear_operator<Op>
    &&math::linear_subspace_of<Subspace, math::codomain_t<Op>> class projected_op final {
  public:
    using domain_type = math::domain_t<Op>;
    using codomain_type = math::codomain_t<Op>;

    projected_op(const Op &op, Subspace subspace) : op_(&op), subspace_(std::move(subspace)) {}

    void apply(const domain_type &x, codomain_type &y) const {
        math::apply(*op_, x, y);
        math::project(subspace_, y);
    }

    [[nodiscard]] auto rows() const { return op_->rows(); }
    [[nodiscard]] auto cols() const { return op_->cols(); }
    [[nodiscard]] const Op &base() const noexcept { return *op_; }
    [[nodiscard]] const Subspace &subspace() const noexcept { return subspace_; }

  private:
    const Op *op_;
    Subspace subspace_;
};

template <class Op, class Subspace>
[[nodiscard]] auto projected(const Op &op, Subspace subspace) {
    return projected_op<Op, Subspace>(op, std::move(subspace));
}

template <class Op, class Subspace>
requires(!std::is_lvalue_reference_v<Op>) auto projected(Op &&, Subspace) = delete;

} // namespace num::operators

namespace num::math {

/// A projection carries its operand's law over to the subspace rather than discarding it.
///
/// \f$P A\f$ does not satisfy a global law even when \f$A\f$ does: \f$(PA)^* = AP \ne
/// PA\f$. It satisfies the law *on the subspace*, because \f$PAx = PAPx\f$ for every
/// \f$x \in S\f$, and \f$PAP\f$ is self-adjoint, semidefinite or definite exactly when
/// \f$A\f$ is. `restricted_to` performs that translation.
///
/// This is what lets `num::pcg` run on a graph Laplacian restricted to the zero-sum
/// subspace without the caller re-asserting definiteness that was already established.
template <class Op, class Subspace>
struct claims_of<operators::projected_op<Op, Subspace>> {
  private:
    template <class... Ls>
    static auto derive(type_list<Ls...>)
        -> type_list<law::restricted_to_t<Ls, Subspace>...>;

  public:
    using type = decltype(derive(typename detail::declared_laws<Op>::type{}));
};

} // namespace num::math
