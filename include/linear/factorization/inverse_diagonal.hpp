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
struct InverseDiagonalWorkspace {
    Matrix right_hand_sides; ///< Reused block of selected identity columns.
    Matrix solutions;        ///< Reused solution block.
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
void inverse_diagonal(const LUResult &factor, Vector &result, InverseDiagonalWorkspace &workspace,
                      idx block_size = 64);
/// Compute diag(A^-1) from a Cholesky factor using blocked identity solves.
void inverse_diagonal(const CholeskyResult &factor, Vector &result,
                      InverseDiagonalWorkspace &workspace, idx block_size = 64);
/// Compute diag(A^-1) from a KLU factor using blocked identity solves.
void inverse_diagonal(const KLUFactor &factor, Vector &result, InverseDiagonalWorkspace &workspace,
                      idx block_size = 64);
/// Compute diag(A^-1) from an automatically selected reusable factor.
void inverse_diagonal(const AutoLinearSolver &factor, Vector &result,
                      InverseDiagonalWorkspace &workspace, idx block_size = 64);

/// Compute A^-1(rows[i], columns[i]) using only the requested inverse columns.
void selected_inverse(const LUResult &factor, std::span<const idx> rows,
                      std::span<const idx> columns, Vector &result,
                      InverseDiagonalWorkspace &workspace);
/// Compute selected inverse entries from a Cholesky factor.
void selected_inverse(const CholeskyResult &factor, std::span<const idx> rows,
                      std::span<const idx> columns, Vector &result,
                      InverseDiagonalWorkspace &workspace);
/// Compute selected inverse entries from a KLU factor.
void selected_inverse(const KLUFactor &factor, std::span<const idx> rows,
                      std::span<const idx> columns, Vector &result,
                      InverseDiagonalWorkspace &workspace);
/// Compute selected inverse entries from an automatically selected factor.
void selected_inverse(const AutoLinearSolver &factor, std::span<const idx> rows,
                      std::span<const idx> columns, Vector &result,
                      InverseDiagonalWorkspace &workspace);

/// Extract A^-1(indices, indices), preserving the requested index order.
void inverse_principal_block(const LUResult &factor, std::span<const idx> indices, Matrix &result,
                             InverseDiagonalWorkspace &workspace);
/// Extract a principal inverse block from a Cholesky factor.
void inverse_principal_block(const CholeskyResult &factor, std::span<const idx> indices,
                             Matrix &result, InverseDiagonalWorkspace &workspace);
/// Extract a principal inverse block from a KLU factor.
void inverse_principal_block(const KLUFactor &factor, std::span<const idx> indices, Matrix &result,
                             InverseDiagonalWorkspace &workspace);
/// Extract a principal inverse block from an automatically selected factor.
void inverse_principal_block(const AutoLinearSolver &factor, std::span<const idx> indices,
                             Matrix &result, InverseDiagonalWorkspace &workspace);

/// Convenience overload owning its result and workspace.
/// Allocates; prefer the workspace form in hot loops.
[[nodiscard]] inline Vector inverse_diagonal(const AutoLinearSolver &factor, idx block_size = 64) {
    Vector result(factor.size(), 0.0);
    InverseDiagonalWorkspace workspace;
    inverse_diagonal(factor, result, workspace, block_size);
    return result;
}

/// Convenience overload owning its result and workspace.
/// Allocates; prefer the workspace form in hot loops.
[[nodiscard]] inline Matrix inverse_principal_block(const AutoLinearSolver &factor,
                                                    std::span<const idx> indices) {
    Matrix result;
    InverseDiagonalWorkspace workspace;
    inverse_principal_block(factor, indices, result, workspace);
    return result;
}

} // namespace num
