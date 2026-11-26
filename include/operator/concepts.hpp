/// @file operator/concepts.hpp
/// @brief Operator concepts that depend on the containers.
#pragma once

#include "container/concepts.hpp"
#include "container/vector.hpp"
#include "core/math/concepts.hpp"
#include <type_traits>

namespace num {

/// @brief Operator that can produce its own CSR matrix.
template <class Op>
concept sparse_convertible = linear_operator<Op> && requires(const Op &A) {
    { A.to_sparse() };
};

} // namespace num
