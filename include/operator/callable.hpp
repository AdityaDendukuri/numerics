/// @file operator/callable.hpp
/// @brief Callable adapter for matrix-free linear operators.
#pragma once

#include "container/vector.hpp"
#include "core/math/associated.hpp"
#include "core/math/models.hpp"
#include <utility>

namespace num::operators {

/// @brief Adapt any callable void(const Vector&, Vector&) to the operator
/// protocol.
template <class F>
struct CallableOp final {
    using domain_type = Vector;
    using codomain_type = Vector;

    /// Adapt a callable to a rectangular operator with explicit dimensions.
    CallableOp(F f, idx rows, idx cols) : f_(std::move(f)), rows_(rows), cols_(cols) {}

    /// Adapt a callable to a square n-by-n operator.
    CallableOp(F f, idx n) : CallableOp(std::move(f), n, n) {}

    /// Evaluate y=A*x, resizing y to the declared row count when needed.
    void apply(const Vector &x, Vector &y) const {
        if (y.size() != rows_) {
            y = Vector(rows_);
        }
        f_(x, y);
    }

    [[nodiscard]] idx rows() const noexcept { return rows_; }
    [[nodiscard]] idx cols() const noexcept { return cols_; }

  private:
    F f_;
    idx rows_;
    idx cols_;
};

template <class F>
/// Construct a rectangular callable operator with inferred callable type.
[[nodiscard]] CallableOp<F> make_op(F f, idx rows, idx cols) {
    return CallableOp<F>(std::move(f), rows, cols);
}

template <class F>
/// Construct a square callable operator with inferred callable type.
[[nodiscard]] CallableOp<F> make_op(F f, idx n) {
    return CallableOp<F>(std::move(f), n);
}

} // namespace num::operators

namespace num::math {

template<class F>
struct model_of<operators::CallableOp<F>> {
    using laws = type_list<law::linear_map>;
};

} // namespace num::math
