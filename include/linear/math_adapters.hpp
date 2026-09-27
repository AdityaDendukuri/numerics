/// @file math_adapters.hpp
/// @brief Dense matrices as linear operators: their action and their spaces.
#pragma once

#include "core/math/operations.hpp"
#include "kernel/kernel.hpp"
#include "container/matrix.hpp"
#include "container/vector.hpp"
#include <stdexcept>

namespace num {

/// Dense storage participates in the map protocol without pretending that every
/// matrix value is positive definite or even square.
template <std::floating_point T>
inline void tag_invoke(math::apply_t, const basic_mat<T> &matrix, const basic_vec<T> &x,
                       basic_vec<T> &y) {
    if (x.size() != matrix.cols()) {
        throw std::invalid_argument("math::apply: dense matrix input dimension mismatch");
    }
    if (y.size() != matrix.rows()) {
        y = basic_vec<T>(matrix.rows());
    }
    kernel::matvec(y.data(), matrix.data(), x.data(), matrix.rows(), matrix.cols());
}

} // namespace num

namespace num::math::detail {

/// A dense matrix maps vectors to vectors of its own scalar.
template <std::floating_point T>
struct domain_of<basic_mat<T>> {
    using type = basic_vec<T>;
};

template <std::floating_point T>
struct codomain_of<basic_mat<T>> {
    using type = basic_vec<T>;
};

} // namespace num::math::detail
