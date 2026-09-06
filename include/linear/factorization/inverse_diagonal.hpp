/// @file linear/factorization/inverse_diagonal.hpp
/// @brief Diagonal of an inverse from reusable factorizations.
#pragma once

#include "linear/factorization/cholesky.hpp"
#include "linear/factorization/lu.hpp"
#include "linear/solvers/auto_linear.hpp"
#include "linear/sparse/klu.hpp"
#include <span>

namespace num {

/// Reusable dense buffers for blocked and selected inverse extraction.
struct inverse_diagonal_workspace {
    mat right_hand_sides; ///< Reused block of selected identity columns.
    mat solutions;        ///< Reused solution block.
};

/// @brief Perform x += y, track accumulated floating-point error, and verify precision acceptability.
template <class T>
inline bool safe_add(T &x, T &err, T y, T tolerance = 1e-6) {
    x += y;
    if (x <= 0) {
        return false;
    }
    err += std::numeric_limits<T>::epsilon() * std::abs(x);
    return (err / x) < tolerance;
}

/// Compute diag(A^-1) through blocked identity solves.
void inverse_diagonal(const lu_result &factor, vec &result, inverse_diagonal_workspace &workspace,
                      idx block_size = 64);
/// Compute diag(A^-1) from a Cholesky factor using blocked identity solves.
void inverse_diagonal(const cholesky_result &factor, vec &result,
                      inverse_diagonal_workspace &workspace, idx block_size = 64);
/// Compute diag(A^-1) from a KLU factor using blocked identity solves.
void inverse_diagonal(const klu_factorization &factor, vec &result, inverse_diagonal_workspace &workspace,
                      idx block_size = 64);
/// Compute diag(A^-1) from an automatically selected reusable factor.
void inverse_diagonal(const auto_linear_solver &factor, vec &result,
                      inverse_diagonal_workspace &workspace, idx block_size = 64);

/// Compute A^-1(rows[i], columns[i]) using only the requested inverse columns.
void selected_inverse(const lu_result &factor, std::span<const idx> rows,
                      std::span<const idx> columns, vec &result,
                      inverse_diagonal_workspace &workspace);
/// Compute selected inverse entries from a Cholesky factor.
void selected_inverse(const cholesky_result &factor, std::span<const idx> rows,
                      std::span<const idx> columns, vec &result,
                      inverse_diagonal_workspace &workspace);
/// Compute selected inverse entries from a KLU factor.
void selected_inverse(const klu_factorization &factor, std::span<const idx> rows,
                      std::span<const idx> columns, vec &result,
                      inverse_diagonal_workspace &workspace);
/// Compute selected inverse entries from an automatically selected factor.
void selected_inverse(const auto_linear_solver &factor, std::span<const idx> rows,
                      std::span<const idx> columns, vec &result,
                      inverse_diagonal_workspace &workspace);

/// Extract A^-1(indices, indices), preserving the requested index order.
void inverse_principal_block(const lu_result &factor, std::span<const idx> indices, mat &result,
                             inverse_diagonal_workspace &workspace);
/// Extract a principal inverse block from a Cholesky factor.
void inverse_principal_block(const cholesky_result &factor, std::span<const idx> indices,
                             mat &result, inverse_diagonal_workspace &workspace);
/// Extract a principal inverse block from a KLU factor.
void inverse_principal_block(const klu_factorization &factor, std::span<const idx> indices, mat &result,
                             inverse_diagonal_workspace &workspace);
/// Extract a principal inverse block from an automatically selected factor.
void inverse_principal_block(const auto_linear_solver &factor, std::span<const idx> indices,
                             mat &result, inverse_diagonal_workspace &workspace);

/// Convenience overload owning its result and workspace.
/// Allocates; prefer the workspace form in hot loops.
[[nodiscard]] inline vec inverse_diagonal(const auto_linear_solver &factor, idx block_size = 64) {
    vec result(factor.size(), 0.0);
    inverse_diagonal_workspace workspace;
    inverse_diagonal(factor, result, workspace, block_size);
    return result;
}

/// Convenience overload owning its result and workspace.
/// Allocates; prefer the workspace form in hot loops.
[[nodiscard]] inline mat inverse_principal_block(const auto_linear_solver &factor,
                                                    std::span<const idx> indices) {
    mat result;
    inverse_diagonal_workspace workspace;
    inverse_principal_block(factor, indices, result, workspace);
    return result;
}

} // namespace num
