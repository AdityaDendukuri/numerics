/// @file linear/sparse/sparse_op.hpp
/// @brief Sparse matrix operator adapter.
#pragma once

#include "core/math/associated.hpp"
#include "core/math/laws.hpp"
#include "linear/sparse/sparse.hpp"
#include "operator/concepts.hpp"
#include <stdexcept>

namespace num::operators {

/// @brief Sparse operator: \f$y \leftarrow A x\f$ for an `spmat`.
struct sparse_op final {
    using domain_type = vec<real>;
    using codomain_type = vec<real>;

    /// @brief Non-owning view of `A`.
    /// @warning `A` must outlive the operator.
    explicit sparse_op(const spmat &A) : A_(A) {}

    /// @brief Operator application: \f$y \leftarrow A x\f$.
    void apply(const vec<real> &x, vec<real> &y) const;
    [[nodiscard]] idx rows() const noexcept { return A_.n_rows(); }
    [[nodiscard]] idx cols() const noexcept { return A_.n_cols(); }

  private:
    const spmat &A_;
};

static_assert(linear_operator<sparse_op>);

inline void sparse_op::apply(const vec<real> &x, vec<real> &y) const {
    if (x.size() != A_.n_cols()) {
        throw std::invalid_argument("sparse_op::apply: input dimension mismatch");
    }
    if (y.size() != A_.n_rows()) {
        y = vec<real>(A_.n_rows());
    }
    sparse_matvec(A_, x, y);
}

} // namespace num::operators

namespace num::math {


} // namespace num::math
