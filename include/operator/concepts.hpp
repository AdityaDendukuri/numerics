/// @file operator/concepts.hpp
/// @brief Operator concepts that need the container tier. The rest are in
/// `core/math/concepts.hpp`.
#pragma once

#include "container/concepts.hpp"
#include "container/vector.hpp"
#include "core/math/concepts.hpp"
#include <type_traits>

namespace num {

/// @brief Linear operator that can materialize itself as explicit sparse CSR storage.
///
/// A statement about representation rather than mathematics: it says an operator can hand
/// over its entries, which a factorization needs and a matrix-free operator cannot do.
///
/// @tparam Op Operator type.
template <class Op>
concept sparse_convertible = linear_operator<Op> && requires(const Op &A) {
    { A.to_sparse() };
};

} // namespace num
