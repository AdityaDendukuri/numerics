/// @file linear/factorization/inverse_diagonal.hpp
/// @brief Inverse diagonals and selected inverse entries from factorizations.
#pragma once

#include "core/types.hpp"
#include "linear/factorization/cholesky.hpp"
#include "linear/factorization/lu.hpp"
#include "linear/solvers/auto_linear.hpp"
#include "linear/sparse/klu.hpp"
#include <span>

namespace num {

/// @brief Reusable buffers for inverse extraction.
struct inverse_diagonal_workspace {
    mat<real> right_hand_sides; ///< Reused block of selected identity columns.
    mat<real> solutions;        ///< Reused solution block.
};

/// @brief Checked accumulation: \f$x \leftarrow x + y\f$, tracking rounding error.
template <class T>
inline bool safe_add(T &x, T &err, T y, T tolerance = 1e-6) {
    x += y;
    if (x <= 0) {
        return false;
    }
    err += std::numeric_limits<T>::epsilon() * std::abs(x);
    return (err / x) < tolerance;
}

/// @brief Inverse diagonal from LU: \f$\mathrm{diag}(A^{-1})\f$.
void inverse_diagonal(const lu_result<real> &factor, vec<real> &result, inverse_diagonal_workspace &workspace,
                      idx block_size = 64);
/// @brief Inverse diagonal from Cholesky: \f$\mathrm{diag}(A^{-1})\f$.
void inverse_diagonal(const cholesky_result &factor, vec<real> &result,
                      inverse_diagonal_workspace &workspace, idx block_size = 64);
/// @brief Inverse diagonal from KLU: \f$\mathrm{diag}(A^{-1})\f$.
void inverse_diagonal(const klu_factorization &factor, vec<real> &result, inverse_diagonal_workspace &workspace,
                      idx block_size = 64);
/// @brief Inverse diagonal from an automatic factor: \f$\mathrm{diag}(A^{-1})\f$.
void inverse_diagonal(const auto_linear_solver &factor, vec<real> &result,
                      inverse_diagonal_workspace &workspace, idx block_size = 64);

/// @brief Selected inverse from LU: \f$(A^{-1})_{r_i c_i}\f$.
void selected_inverse(const lu_result<real> &factor, view<const idx> rows,
                      view<const idx> columns, vec<real> &result,
                      inverse_diagonal_workspace &workspace);
/// @brief Selected inverse from Cholesky: \f$(A^{-1})_{r_i c_i}\f$.
void selected_inverse(const cholesky_result &factor, view<const idx> rows,
                      view<const idx> columns, vec<real> &result,
                      inverse_diagonal_workspace &workspace);
/// @brief Selected inverse from KLU: \f$(A^{-1})_{r_i c_i}\f$.
void selected_inverse(const klu_factorization &factor, view<const idx> rows,
                      view<const idx> columns, vec<real> &result,
                      inverse_diagonal_workspace &workspace);
/// @brief Selected inverse from an automatic factor: \f$(A^{-1})_{r_i c_i}\f$.
void selected_inverse(const auto_linear_solver &factor, view<const idx> rows,
                      view<const idx> columns, vec<real> &result,
                      inverse_diagonal_workspace &workspace);

/// @brief Principal inverse block from LU: \f$(A^{-1})_{I I}\f$.
void inverse_principal_block(const lu_result<real> &factor, view<const idx> indices, mat<real> &result,
                             inverse_diagonal_workspace &workspace);
/// @brief Principal inverse block from Cholesky: \f$(A^{-1})_{I I}\f$.
void inverse_principal_block(const cholesky_result &factor, view<const idx> indices,
                             mat<real> &result, inverse_diagonal_workspace &workspace);
/// @brief Principal inverse block from KLU: \f$(A^{-1})_{I I}\f$.
void inverse_principal_block(const klu_factorization &factor, view<const idx> indices, mat<real> &result,
                             inverse_diagonal_workspace &workspace);
/// @brief Principal inverse block from an automatic factor: \f$(A^{-1})_{I I}\f$.
void inverse_principal_block(const auto_linear_solver &factor, view<const idx> indices,
                             mat<real> &result, inverse_diagonal_workspace &workspace);

/// @brief Inverse diagonal: \f$\mathrm{diag}(A^{-1})\f$. Allocates.
[[nodiscard]] inline vec<real> inverse_diagonal(const auto_linear_solver &factor, idx block_size = 64) {
    vec<real> result(factor.size(), 0.0);
    inverse_diagonal_workspace workspace;
    inverse_diagonal(factor, result, workspace, block_size);
    return result;
}

/// @brief Principal inverse block: \f$(A^{-1})_{I I}\f$. Allocates.
[[nodiscard]] inline mat<real> inverse_principal_block(const auto_linear_solver &factor,
                                                    view<const idx> indices) {
    mat<real> result;
    inverse_diagonal_workspace workspace;
    inverse_principal_block(factor, indices, result, workspace);
    return result;
}

} // namespace num
