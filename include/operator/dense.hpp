/// @file operator/dense.hpp
/// @brief Dense mat adapter for the operator protocol.
#pragma once

#include "container/matrix_ops.hpp"
#include "core/math/associated.hpp"
#include "core/math/models.hpp"
#include <stdexcept>
#include "container/matrix.hpp"
#include "core/policy.hpp"
#include "operator/concepts.hpp"

namespace num::operators {

/// @brief Adapt a dense mat to the operator protocol.
struct dense_op final {
    using math_laws = math::type_list<law::linear_map>;
    using domain_type = vec;
    using codomain_type = vec;

    /// Store a non-owning matrix reference.
    explicit dense_op(const mat &A) : A_(A) {}

    /// Compute y=A*x using `num::accel` (the build's best available backend).
    void apply(const vec &x, vec &y) const;

    /// Compute \f$x = A^T y\f$, the adjoint over \f$\mathbb{R}\f$.
    ///
    /// Without this, `dense_op` is not an @ref num::math::adjointable_linear_operator,
    /// which nothing in the library satisfied before.
    void apply_adjoint(const vec &y, vec &x) const;

    [[nodiscard]] idx rows() const noexcept { return A_.rows(); }
    [[nodiscard]] idx cols() const noexcept { return A_.cols(); }

  private:
    const mat &A_;
};

static_assert(linear_operator<dense_op>);
static_assert(adjointable_linear_operator<dense_op>);



inline void dense_op::apply(const vec &x, vec &y) const {
    if (x.size() != A_.cols()) {
        throw std::invalid_argument("dense_op::apply: input dimension mismatch");
    }
    if (y.size() != A_.rows()) {
        y = vec(A_.rows());
    }
    matvec(A_, x, y);
}

inline void dense_op::apply_adjoint(const vec &y, vec &x) const {
    if (y.size() != A_.rows()) {
        throw std::invalid_argument("dense_op::apply_adjoint: input dimension mismatch");
    }
    if (x.size() != A_.cols()) {
        x = vec(A_.cols());
    }
    kernel::matvec_transpose(x.data(), A_.data(), y.data(), A_.rows(), A_.cols());
}

} // namespace num::operators

namespace num::math {


} // namespace num::math
