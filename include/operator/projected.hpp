/// @file operator/projected.hpp
/// @brief Projected operators: \f$P_S A\f$.
#pragma once

#include "core/math/associated.hpp"
#include "core/math/concepts.hpp"
#include "core/math/subspace.hpp"
#include <type_traits>
#include <utility>

namespace num::operators {

/// @brief Law of \f$P_S A\f$ on \f$S\f$: the law of \f$A\f$, restricted to \f$S\f$.
template <class Op, class Subspace>
using restricted_laws = std::conditional_t<
    claims<Op, law::spd>, law::list<law::spd_on<Subspace>>,
    std::conditional_t<claims<Op, law::psd>, law::list<law::psd_on<Subspace>>,
                       std::conditional_t<claims<Op, law::self_adjoint>,
                                          law::list<law::self_adjoint_on<Subspace>>, law::list<>>>>;

/// @brief Operator followed by a projection onto a subspace:
/// \f$y \leftarrow P_S A x\f$. Holds references.
/// @warning Holds `A` by reference; it must outlive the projected operator.
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

/// @brief Project an operator onto a subspace, carrying its law over to \f$S\f$: \f$P_S A\f$.
template <class Op, class Subspace>
[[nodiscard]] auto projected(const Op &op, Subspace subspace) {
    return projected_op<Op, Subspace>(op, std::move(subspace));
}

template <class Op, class Subspace>
requires(!std::is_lvalue_reference_v<Op>) auto projected(Op &&, Subspace) = delete;

} // namespace num::operators

