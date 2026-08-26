/// @file linear/sparse/sparse_op.hpp
/// @brief SparseMatrix adapter for the operator protocol.
///
/// Lives beside SparseMatrix (rather than under operator/) so the operator
/// module stays free of any linear-algebra dependency; the LinearOperator contract it
/// models is defined in operator/concepts.hpp.
#pragma once

#include "core/math/associated.hpp"
#include "core/math/models.hpp"
#include "linear/sparse/sparse.hpp"
#include "operator/concepts.hpp"
#include <stdexcept>

namespace num::operators {

/// @brief Adapt a SparseMatrix to the operator protocol.
struct SparseOp final {
    using domain_type = Vector;
    using codomain_type = Vector;

    /// Store a non-owning reference to a CSR matrix.
    explicit SparseOp(const SparseMatrix &A) : A_(A) {}

    /// Compute y=A*x.
    void apply(const Vector &x, Vector &y) const;
    [[nodiscard]] idx rows() const noexcept { return A_.n_rows(); }
    [[nodiscard]] idx cols() const noexcept { return A_.n_cols(); }

  private:
    const SparseMatrix &A_;
};

static_assert(LinearOperator<SparseOp>);

inline void SparseOp::apply(const Vector &x, Vector &y) const {
    if (x.size() != A_.n_cols()) {
        throw std::invalid_argument("SparseOp::apply: input dimension mismatch");
    }
    if (y.size() != A_.n_rows()) {
        y = Vector(A_.n_rows());
    }
    sparse_matvec(A_, x, y);
}

} // namespace num::operators

namespace num::math {

template <>
struct model_of<operators::SparseOp> {
    using laws = type_list<law::linear_map>;
};

} // namespace num::math
