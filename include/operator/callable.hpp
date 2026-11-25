/// @file operator/callable.hpp
/// @brief Matrix-free operators from callables.
#pragma once

#include "container/vector.hpp"
#include "core/math/associated.hpp"
#include "core/math/laws.hpp"
#include <utility>

namespace num::operators {

/// @brief Callable operator: \f$y \leftarrow f(x)\f$ for `f(const vec &, vec &)`.
template <class F>
struct callable_op final {
    using domain_type = vec<real>;
    using codomain_type = vec<real>;

    /// @brief Rectangular callable operator, `rows x cols`.
    callable_op(F f, idx rows, idx cols) : f_(std::move(f)), rows_(rows), cols_(cols) {}

    /// @brief Square callable operator, `n x n`.
    callable_op(F f, idx n) : callable_op(std::move(f), n, n) {}

    /// @brief Operator application: \f$y \leftarrow A x\f$; resizes `y` if needed.
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
/// @brief Rectangular callable operator, type deduced.
[[nodiscard]] callable_op<F> make_op(F f, idx rows, idx cols) {
    return callable_op<F>(std::move(f), rows, cols);
}

template <class F>
/// @brief Square callable operator, type deduced.
[[nodiscard]] callable_op<F> make_op(F f, idx n) {
    return callable_op<F>(std::move(f), n);
}

} // namespace num::operators

