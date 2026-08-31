/// @file operator/dense.hpp
/// @brief Dense Matrix adapter for the operator protocol.
#pragma once

#include "container/matrix_ops.hpp"
#include "core/math/associated.hpp"
#include "core/math/models.hpp"
#include <stdexcept>
#include "container/matrix.hpp"
#include "core/policy.hpp"
#include "operator/concepts.hpp"

namespace num::operators {

/// @brief Adapt a dense Matrix to the operator protocol.
struct DenseOp final {
    using domain_type = Vector;
    using codomain_type = Vector;

    /// Store a non-owning matrix reference and backend selection.
    explicit DenseOp(const Matrix &A, Backend b = backend::dflt) : A_(A), b_(b) {}

    /// Compute y=A*x using the selected backend.
    void apply(const Vector &x, Vector &y) const;
    [[nodiscard]] idx rows() const noexcept { return A_.rows(); }
    [[nodiscard]] idx cols() const noexcept { return A_.cols(); }

  private:
    const Matrix &A_;
    Backend b_;
};

static_assert(LinearOperator<DenseOp>);



inline void DenseOp::apply(const Vector &x, Vector &y) const {
    if (x.size() != A_.cols()) {
        throw std::invalid_argument("DenseOp::apply: input dimension mismatch");
    }
    if (y.size() != A_.rows()) {
        y = Vector(A_.rows());
    }
    matvec(A_, x, y, b_);
}

} // namespace num::operators

namespace num::math {

template<>
struct model_of<operators::DenseOp> {
    using laws = type_list<law::linear_map>;
};

} // namespace num::math
