/// @file projected.hpp
/// @brief Operator adapter that projects every output onto a linear subspace.
#pragma once

#include "core/math/associated.hpp"
#include "core/math/concepts.hpp"
#include "core/math/subspace.hpp"
#include <type_traits>
#include <utility>

namespace num::operators {

/// @brief The law of \f$P_S A\f$ on S for an operator A claiming a self-adjoint law.
///
/// \f$P_S A\f$ is not self-adjoint even when A is, but \f$P_S A x = P_S A P_S x\f$ for
/// \f$x \in S\f$, and \f$P_S A P_S\f$ keeps A's law on S. This lets `num::pcg` solve with a
/// graph Laplacian on the zero-sum subspace without a second assertion.
template <class Op, class Subspace>
using restricted_laws = std::conditional_t<
    claims<Op, law::spd>, law::list<law::spd_on<Subspace>>,
    std::conditional_t<claims<Op, law::psd>, law::list<law::psd_on<Subspace>>,
                       std::conditional_t<claims<Op, law::self_adjoint>,
                                          law::list<law::self_adjoint_on<Subspace>>, law::list<>>>>;

/// Non-owning representation of P_S A, where P_S is projection onto S.
template <class Op, class Subspace>
requires math::linear_operator<Op>
    &&math::linear_subspace_of<Subspace, math::codomain_t<Op>> class projected_op final {
  public:
    using laws = restricted_laws<Op, Subspace>;
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

/// @brief \f$P_S A\f$, which carries the law of `A` onto the subspace `S`. It holds `A` by
/// reference.
template <class Op, class Subspace>
[[nodiscard]] auto projected(const Op &op, Subspace subspace) {
    return projected_op<Op, Subspace>(op, std::move(subspace));
}

template <class Op, class Subspace>
requires(!std::is_lvalue_reference_v<Op>) auto projected(Op &&, Subspace) = delete;

} // namespace num::operators

