/// @file linear/debug.hpp
/// @brief Runtime validation of structural invariants carried by stored matrices.
///
/// These are the diagnostic siblings of the structural concepts in
/// linear/concepts.hpp. Unlike the operator axioms, which can only ever be
/// sampled, everything here is *decidable*: whether a CSR index array is
/// monotonic, whether a matrix is square, whether a bandwidth fits its dimension.
/// They are checked exhaustively rather than probed.
#pragma once

#include "core/debug.hpp"
#include "container/concepts.hpp"
#include "core/types.hpp"
#include <cmath>
#include <source_location>
#include <string>

namespace num::linear::debug {

using num::debug::check_dim;
using num::debug::check_non_empty;
using num::debug::diagnostic_level;
using num::debug::get_level;
using num::debug::panic;

/// @brief Validate structural invariants of CSR sparse storage.
template <class SparseType>
inline void verify_sparse_structure(const SparseType &A,
                                    std::source_location loc = std::source_location::current()) {
    if (get_level() == diagnostic_level::off) {
        return;
    }
    const idx nrows = A.n_rows();
    const idx ncols = A.n_cols();
    const idx *row_ptr = A.row_ptr();
    const idx *col_idx = A.col_idx();
    const real *values = A.values();

    if (row_ptr[0] != 0) {
        panic("SparseStructureError", "row_ptr[0] must be 0", loc);
    }
    for (idx i = 0; i < nrows; ++i) {
        if (row_ptr[i] > row_ptr[i + 1]) {
            panic("SparseStructureError",
                  "row_ptr is not monotonic at row " + std::to_string(i), loc);
        }
        for (idx k = row_ptr[i]; k < row_ptr[i + 1]; ++k) {
            if (col_idx[k] >= ncols) {
                panic("SparseStructureError",
                      "col_idx[" + std::to_string(k) + "] = " + std::to_string(col_idx[k]) +
                          " exceeds n_cols (" + std::to_string(ncols) + ")",
                      loc);
            }
            if (!std::isfinite(values[k])) {
                panic("SparseStructureError",
                      "non-finite sparse value at index " + std::to_string(k), loc);
            }
        }
    }
}

} // namespace num::linear::debug
