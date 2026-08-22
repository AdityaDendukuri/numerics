/// @file core/debug.hpp
/// @brief Runtime contract validation and Julia/NumPy-style diagnostic checks.
#pragma once

#include "core/types.hpp"
#include <algorithm>
#include <cmath>
#include <concepts>
#include <iostream>
#include <source_location>
#include <stdexcept>
#include <string>
#include <string_view>

namespace num::debug {

enum class DiagnosticLevel {
    off = 0,
    basic = 1, ///< Dimension matching, non-empty, finite checks
    full = 2   ///< Includes sampled symmetry, adjoint, and positive-definiteness testing
};

inline DiagnosticLevel g_level = DiagnosticLevel::full;

inline void set_level(DiagnosticLevel lvl) noexcept {
    g_level = lvl;
}

inline DiagnosticLevel get_level() noexcept {
    return g_level;
}

/// @brief Raise a descriptive diagnostic exception with source location info.
[[noreturn]] inline void panic(std::string_view category, std::string_view message,
                               std::source_location loc = std::source_location::current()) {
    std::string err = "[" + std::string(category) + "] Error at " + loc.file_name() + ":" +
                      std::to_string(loc.line()) + " in " + loc.function_name() + ":\n  " +
                      std::string(message);
    throw std::invalid_argument(err);
}

/// @brief Verify dimension equality (e.g. A.rows() == b.size())
inline void check_dim(idx expected, idx actual, std::string_view label,
                      std::source_location loc = std::source_location::current()) {
    if (g_level == DiagnosticLevel::off) {
        return;
    }
    if (expected != actual) {
        std::string msg = "Dimension mismatch for " + std::string(label) + ": expected " +
                          std::to_string(expected) + ", got " + std::to_string(actual);
        panic("DimensionError", msg, loc);
    }
}

/// @brief Verify a container is non-empty
inline void check_non_empty(idx size, std::string_view label,
                            std::source_location loc = std::source_location::current()) {
    if (g_level == DiagnosticLevel::off) {
        return;
    }
    if (size == 0) {
        panic("ValueError", std::string(label) + " cannot be empty (size is 0)", loc);
    }
}

/// @brief Verify all values in array are finite (not NaN or Inf)
template <typename T>
inline void check_finite(const T *data, idx n, std::string_view label,
                         std::source_location loc = std::source_location::current()) {
    if (g_level == DiagnosticLevel::off) {
        return;
    }
    for (idx i = 0; i < n; ++i) {
        if (!std::isfinite(data[i])) {
            panic("ValueError",
                  std::string(label) + " contains non-finite value (NaN or Inf) at index " +
                      std::to_string(i),
                  loc);
        }
    }
}

/// @brief Sampled runtime test for operator positive-definiteness (x^T A x > 0)
template <class Op, class VectorType>
inline void verify_spd_sample(const Op &A, idx n,
                              std::source_location loc = std::source_location::current()) {
    if (g_level != DiagnosticLevel::full) {
        return;
    }

    if (n == 0) {
        return;
    }
    VectorType x(n, real(1.0));
    VectorType ax(n, real(0.0));
    A.apply(x, ax);

    real dot_val = 0.0;
    for (idx i = 0; i < n; ++i) {
        dot_val += x[i] * ax[i];
    }

    if (dot_val <= 0.0) {
        panic("PropertyError",
              "assume_spd() assertion failed: sampled inner product x^T A x = " +
                  std::to_string(dot_val) + " <= 0. The operator is NOT positive definite!",
              loc);
    }
}

/// @brief Sampled runtime test for operator symmetry (x^T A y approx y^T A x)
template <class Op, class VectorType>
inline void verify_symmetry_sample(const Op &A, idx n,
                                   std::source_location loc = std::source_location::current()) {
    if (g_level != DiagnosticLevel::full) {
        return;
    }

    if (n <= 1) {
        return;
    }
    VectorType x(n), y(n), Ax(n), Ay(n);
    for (idx i = 0; i < n; ++i) {
        x[i] = (i % 2 == 0) ? real(1.0) : real(0.5);
        y[i] = (i % 3 == 0) ? real(0.7) : real(1.3);
    }
    A.apply(x, Ax);
    A.apply(y, Ay);

    real xAy = 0.0, yAx = 0.0;
    for (idx i = 0; i < n; ++i) {
        xAy += x[i] * Ay[i];
        yAx += y[i] * Ax[i];
    }

    real diff = std::abs(xAy - yAx);
    real scale = std::max(std::abs(xAy), std::abs(yAx)) + 1e-12;
    if (diff / scale > 1e-3) {
        panic("PropertyError",
              "assume_symmetric() assertion failed: sampled |x^T A y - y^T A x| = " +
                  std::to_string(diff) + ". The operator is NOT symmetric!",
              loc);
    }
}

/// @brief Sampled runtime test for adjoint consistency: <A x, y> approx <x, A* y>
template <class Op, class VectorType>
inline void verify_adjoint_sample(const Op &A, idx m, idx n,
                                  std::source_location loc = std::source_location::current()) {
    if (g_level != DiagnosticLevel::full) {
        return;
    }
    if (m == 0 || n == 0) {
        return;
    }
    VectorType x(n), y(m), Ax(m), Aty(n);
    for (idx i = 0; i < n; ++i) {
        x[i] = (i % 2 == 0) ? real(1.0) : real(-0.5);
    }
    for (idx i = 0; i < m; ++i) {
        y[i] = (i % 3 == 0) ? real(0.8) : real(1.2);
    }
    A.apply(x, Ax);
    A.apply_adjoint(y, Aty);

    real dot_ax_y = 0.0, dot_x_aty = 0.0;
    for (idx i = 0; i < m; ++i) {
        dot_ax_y += Ax[i] * y[i];
    }
    for (idx i = 0; i < n; ++i) {
        dot_x_aty += x[i] * Aty[i];
    }

    real diff = std::abs(dot_ax_y - dot_x_aty);
    real scale = std::max(std::abs(dot_ax_y), std::abs(dot_x_aty)) + 1e-12;
    if (diff / scale > 1e-3) {
        panic("PropertyError",
              "Adjoint consistency check failed: |<Ax, y> - <x, A*y>| = " + std::to_string(diff),
              loc);
    }
}

/// @brief Validate structural invariants of CSR sparse storage.
template <class SparseType>
inline void verify_sparse_structure(const SparseType &A,
                                    std::source_location loc = std::source_location::current()) {
    if (g_level == DiagnosticLevel::off) {
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

} // namespace num::debug
