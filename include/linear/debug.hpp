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

/// @brief Validate square dimensions (rows == cols).
template <class MatrixType>
inline void verify_square(const MatrixType &A,
                          std::source_location loc = std::source_location::current()) {
    check_dim(A.rows(), A.cols(), "square matrix dimension", loc);
}

/// @brief Validate tridiagonal dimension invariants.
template <class VecType>
inline void verify_tridiagonal_structure(const VecType &dl, const VecType &d, const VecType &du,
                                         std::source_location loc = std::source_location::current()) {
    if (get_level() == diagnostic_level::off) {
        return;
    }
    const idx n = d.size();
    check_non_empty(n, "main diagonal d", loc);
    check_dim(n - 1, dl.size(), "subdiagonal dl size", loc);
    check_dim(n - 1, du.size(), "superdiagonal du size", loc);
}

/// @brief Validate banded matrix bandwidth bounds.
template <class BandedType>
inline void verify_banded_structure(const BandedType &B,
                                    std::source_location loc = std::source_location::current()) {
    if (get_level() == diagnostic_level::off) {
        return;
    }
    check_dim(B.rows(), B.cols(), "banded matrix dimension", loc);
    const num::idx kl = repr::lower_bandwidth_of(B);
    const num::idx ku = repr::upper_bandwidth_of(B);
    if (kl >= B.rows() || ku >= B.cols()) {
        panic("BandedStructureError",
              "bandwidth exceeds matrix dimension: kl=" + std::to_string(kl) + ", ku=" +
                  std::to_string(ku) + " for a " + std::to_string(B.rows()) + "x" +
                  std::to_string(B.cols()) + " matrix",
              loc);
    }
}


/// @brief Verify that entries outside the declared band actually vanish.
///
/// The dimension check alone only confirms the bandwidths are representable. This
/// is the substantive claim: a banded solver reads nothing outside the band, so a
/// nonzero entry there is silently discarded rather than producing a wrong answer
/// loudly.
template <class BandedType>
inline void verify_band_occupancy(const BandedType &B, double tol = 0.0,
                                  std::source_location loc = std::source_location::current()) {
    if (get_level() == diagnostic_level::off) {
        return;
    }
    const num::idx kl = repr::lower_bandwidth_of(B);
    const num::idx ku = repr::upper_bandwidth_of(B);
    for (num::idx i = 0; i < B.rows(); ++i) {
        for (num::idx j = 0; j < B.cols(); ++j) {
            const bool below = (i > j) && ((i - j) > kl);
            const bool above = (j > i) && ((j - i) > ku);
            if (!below && !above) {
                continue;
            }
            if (std::abs(static_cast<double>(B(i, j))) > tol) {
                panic("BandedStructureError",
                      "entry (" + std::to_string(i) + "," + std::to_string(j) + ") = " +
                          std::to_string(static_cast<double>(B(i, j))) +
                          " lies outside the declared band (kl=" + std::to_string(kl) +
                          ", ku=" + std::to_string(ku) +
                          "), where a banded routine would never read it.",
                      loc);
            }
        }
    }
}
} // namespace num::linear::debug
