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
requires math::LinearOperator<Op>
    &&math::LinearSubspaceOf<Subspace, math::codomain_t<Op>> class ProjectedOp final {
  public:
    using domain_type = math::domain_t<Op>;
    using codomain_type = math::codomain_t<Op>;

    ProjectedOp(const Op &op, Subspace subspace) : op_(&op), subspace_(std::move(subspace)) {}

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
    return ProjectedOp<Op, Subspace>(op, std::move(subspace));
}

template <class Op, class Subspace>
requires(!std::is_lvalue_reference_v<Op>) auto projected(Op &&, Subspace) = delete;

} // namespace num::operators

namespace num::math {

template <class Op, class Subspace>
struct model_of<operators::ProjectedOp<Op, Subspace>> {
    using laws = type_list<law::linear_map>;
};

} // namespace num::math
