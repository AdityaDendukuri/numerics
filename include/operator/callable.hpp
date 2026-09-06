/// @file operator/callable.hpp
/// @brief Callable adapter for matrix-free linear operators.
#pragma once

#include "container/vector.hpp"
#include "core/math/associated.hpp"
#include "core/math/models.hpp"
#include <utility>

namespace num::operators {

/// @brief Adapt any callable void(const vec&, vec&) to the operator
/// protocol.
template <class F>
struct callable_op final {
    using domain_type = vec;
    using codomain_type = vec;

    /// Adapt a callable to a rectangular operator with explicit dimensions.
    callable_op(F f, idx rows, idx cols) : f_(std::move(f)), rows_(rows), cols_(cols) {}

    /// Adapt a callable to a square n-by-n operator.
    callable_op(F f, idx n) : callable_op(std::move(f), n, n) {}

    /// Evaluate y=A*x, resizing y to the declared row count when needed.
    void apply(const vec &x, vec &y) const {
        if (y.size() != rows_) {
            y = vec(rows_);
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
[[nodiscard]] callable_op<F> make_op(F f, idx rows, idx cols) {
    return callable_op<F>(std::move(f), rows, cols);
}

template <class F>
/// Construct a square callable operator with inferred callable type.
[[nodiscard]] callable_op<F> make_op(F f, idx n) {
    return callable_op<F>(std::move(f), n);
}

} // namespace num::operators

namespace num::math {

template<class F>
struct claims_of<operators::callable_op<F>> {
    using type = type_list<law::linear_map>;
};

} // namespace num::math
