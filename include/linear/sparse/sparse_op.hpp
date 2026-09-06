/// @file linear/sparse/sparse_op.hpp
/// @brief spmat adapter for the operator protocol.
///
/// Lives beside spmat (rather than under operator/) so the operator
/// module stays free of any linear-algebra dependency; the linear_operator contract it
/// models is defined in operator/concepts.hpp.
#pragma once

#include "core/math/associated.hpp"
#include "core/math/models.hpp"
#include "linear/sparse/sparse.hpp"
#include "operator/concepts.hpp"
#include <stdexcept>

namespace num::operators {

/// @brief Adapt a spmat to the operator protocol.
struct sparse_op final {
    using math_laws = math::type_list<law::linear_map>;
    using domain_type = vec;
    using codomain_type = vec;

    /// Store a non-owning reference to a CSR matrix.
    explicit sparse_op(const spmat &A) : A_(A) {}

    /// Compute y=A*x.
    void apply(const vec &x, vec &y) const;
    [[nodiscard]] idx rows() const noexcept { return A_.n_rows(); }
    [[nodiscard]] idx cols() const noexcept { return A_.n_cols(); }

  private:
    const spmat &A_;
};

static_assert(linear_operator<sparse_op>);

inline void sparse_op::apply(const vec &x, vec &y) const {
    if (x.size() != A_.n_cols()) {
        throw std::invalid_argument("sparse_op::apply: input dimension mismatch");
    }
    if (y.size() != A_.n_rows()) {
        y = vec(A_.n_rows());
    }
    sparse_matvec(A_, x, y);
}

} // namespace num::operators

namespace num::math {


} // namespace num::math
