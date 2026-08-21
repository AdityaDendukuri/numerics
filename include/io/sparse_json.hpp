/// @file sparse_json.hpp
/// @brief Generic JSON serialization for Numerics sparse matrices.
#pragma once

#include "io/json.hpp"
#include "linalg/sparse/sparse.hpp"

namespace num::io {

[[nodiscard]] SparseMatrix sparse_matrix(const boost::json::value& value);
[[nodiscard]] boost::json::value sparse_matrix_json(const SparseMatrix& matrix);

} // namespace num::io
