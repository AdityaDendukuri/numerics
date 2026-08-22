/// @file sparse_json.hpp
/// @brief Generic JSON serialization for Numerics sparse matrices using nlohmann::json.
#pragma once

#include "io/json.hpp"
#include "linalg/sparse/sparse.hpp"

namespace num::io {

/// Decode a CSR or CSC sparse-matrix JSON object into native CSR storage.
[[nodiscard]] SparseMatrix sparse_matrix(const json &value);
/// Encode a sparse matrix as a CSR JSON object.
[[nodiscard]] json sparse_matrix_json(const SparseMatrix &matrix);

} // namespace num::io
