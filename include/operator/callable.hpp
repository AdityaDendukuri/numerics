/// @file operator/callable.hpp
/// @brief Matrix-free operators from callables.
#pragma once

#include "container/vector.hpp"
#include "core/math/associated.hpp"
#include "core/math/laws.hpp"
#include <utility>

namespace num::operators {

/// @brief Operator defined by a function `f(const vec &, vec &)`: \f$y \leftarrow f(x)\f$.
template <class F>
struct callable_op final {
    using domain_type = vec<real>;
    using codomain_type = vec<real>;

    /// @brief Construct a `rows x cols` operator from a function.
    callable_op(F f, idx rows, idx cols) : f_(std::move(f)), rows_(rows), cols_(cols) {}

    /// @brief Construct an `n x n` operator from a function.
    callable_op(F f, idx n) : callable_op(std::move(f), n, n) {}

    /// @brief Apply the operator, resizing `y` if needed: \f$y \leftarrow A x\f$.
    void apply(const vec<real> &x, vec<real> &y) const {
        if (y.size() != rows_) {
            y = vec<real>(rows_);
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
/// @brief Construct a rectangular operator from a function, deducing its type.
[[nodiscard]] callable_op<F> make_op(F f, idx rows, idx cols) {
    return callable_op<F>(std::move(f), rows, cols);
}

template <class F>
/// @brief Construct a square operator from a function, deducing its type.
[[nodiscard]] callable_op<F> make_op(F f, idx n) {
    return callable_op<F>(std::move(f), n);
}

} // namespace num::operators

