/// @file operator/dense.hpp
/// @brief Dense matrix operator adapter.
#pragma once

#include "container/matrix_ops.hpp"
#include "core/math/associated.hpp"
#include "core/math/laws.hpp"
#include <stdexcept>
#include "container/matrix.hpp"
#include "core/policy.hpp"
#include "operator/concepts.hpp"

namespace num::operators {

/// @brief Operator that applies a dense matrix: \f$y \leftarrow A x\f$.
struct dense_op final {
    using domain_type = vec<real>;
    using codomain_type = vec<real>;

    /// @brief Hold a reference to `A` without copying it.
    /// @warning `A` must outlive the operator.
    explicit dense_op(const mat<real> &A) : A_(A) {}

    /// @brief Apply the matrix: \f$y \leftarrow A x\f$.
    void apply(const vec<real> &x, vec<real> &y) const;

    [[nodiscard]] idx rows() const noexcept { return A_.rows(); }
    [[nodiscard]] idx cols() const noexcept { return A_.cols(); }

  private:
    const mat<real> &A_;
};

static_assert(linear_operator<dense_op>);

inline void dense_op::apply(const vec<real> &x, vec<real> &y) const {
    if (x.size() != A_.cols()) {
        throw std::invalid_argument("dense_op::apply: input dimension mismatch");
    }
    if (y.size() != A_.rows()) {
        y = vec<real>(A_.rows());
    }
    matvec(A_, x, y);
}

} // namespace num::operators

