/// @file linear/factorization/inverse_diagonal.hpp
/// @brief Inverse diagonals and selected inverse entries from any factorization.
#pragma once

#include "core/types.hpp"
#include "linear/factorization/cholesky.hpp"
#include "linear/factorization/lu.hpp"
#include "linear/matrix_utils.hpp"
#include "linear/solve.hpp"
#include <algorithm>
#include <concepts>
#include <stdexcept>
#include <unordered_map>

namespace num {

/// @brief Factorization that also reports its order \f$n\f$ through `size()`.
template <class F>
concept sized_factorization = factorization<F> && requires(const F &factor) {
    { factor.size() } -> std::convertible_to<idx>;
};

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

namespace detail {

template <typename Solve>
inline void compute(idx n, vec<real> &result, inverse_diagonal_workspace &workspace, idx block_size,
             Solve &&solve) {
    if (result.size() != n || block_size == 0) {
        throw std::invalid_argument("inverse_diagonal: invalid result or block size");
    }
    for (idx first = 0; first < n; first += block_size) {
        const idx count = std::min(block_size, n - first);
        workspace.right_hand_sides = identity_columns(n, first, count);
        workspace.solutions = mat<real>(n, count, 0.0);
        solve(workspace.right_hand_sides, workspace.solutions);
        for (idx column = 0; column < count; ++column) {
            result[first + column] = workspace.solutions(first + column, column);
        }
    }
}

template <typename Solve>
inline void compute_selected(idx n, view<const idx> rows, view<const idx> columns,
                      vec<real> &result, inverse_diagonal_workspace &workspace, Solve &&solve) {
    if (rows.size() != columns.size() || result.size() != rows.size()) {
        throw std::invalid_argument("selected_inverse: request sizes must match");
    }
    array<idx> unique_columns;
    unordered_map<idx, idx> position;
    for (idx column : columns) {
        if (column >= n) {
            throw std::out_of_range("selected_inverse: column out of range");
        }
        if (!position.contains(column)) {
            position[column] = unique_columns.size();
            unique_columns.push_back(column);
        }
    }
    workspace.right_hand_sides = mat<real>(n, unique_columns.size(), 0.0);
    for (idx local = 0; local < unique_columns.size(); ++local) {
        workspace.right_hand_sides(unique_columns[local], local) = 1.0;
    }
    workspace.solutions = mat<real>(n, unique_columns.size(), 0.0);
    solve(workspace.right_hand_sides, workspace.solutions);
    for (idx request = 0; request < rows.size(); ++request) {
        if (rows[request] >= n) {
            throw std::out_of_range("selected_inverse: row out of range");
        }
        result[request] = workspace.solutions(rows[request], position[columns[request]]);
    }
}

template <typename Solve>
inline void compute_principal_block(idx n, view<const idx> indices, mat<real> &result,
                             inverse_diagonal_workspace &workspace, Solve &&solve) {
    if (indices.empty()) {
        workspace.right_hand_sides = mat<real>(n, 0, 0.0);
        workspace.solutions = mat<real>(n, 0, 0.0);
        result = mat<real>(0, 0, 0.0);
        return;
    }
    array<bool> seen(n, false);
    workspace.right_hand_sides = mat<real>(n, indices.size(), 0.0);
    for (idx column = 0; column < indices.size(); ++column) {
        const idx index = indices[column];
        if (index >= n) {
            throw std::out_of_range("inverse_principal_block: index out of range");
        }
        if (seen[index]) {
            throw std::invalid_argument("inverse_principal_block: indices must be unique");
        }
        seen[index] = true;
        workspace.right_hand_sides(index, column) = 1.0;
    }

    workspace.solutions = mat<real>(n, indices.size(), 0.0);
    solve(workspace.right_hand_sides, workspace.solutions);
    result = mat<real>(indices.size(), indices.size(), 0.0);
    for (idx row = 0; row < indices.size(); ++row) {
        for (idx column = 0; column < indices.size(); ++column) {
            result(row, column) = workspace.solutions(indices[row], column);
        }
    }
}

} // namespace detail

/// @brief Inverse diagonal: \f$\mathrm{diag}(A^{-1})\f$, by blocks of identity columns.
template <sized_factorization F>
inline void inverse_diagonal(const F &factor, vec<real> &result, inverse_diagonal_workspace &workspace,
                             idx block_size = 64) {
    detail::compute(static_cast<idx>(factor.size()), result, workspace, block_size,
                    [&](const mat<real> &rhs, mat<real> &solution) { solve(factor, rhs, solution); });
}

/// @brief Selected inverse entries: \f$(A^{-1})_{r_i c_i}\f$.
template <sized_factorization F>
inline void selected_inverse(const F &factor, view<const idx> rows, view<const idx> columns,
                             vec<real> &result, inverse_diagonal_workspace &workspace) {
    detail::compute_selected(static_cast<idx>(factor.size()), rows, columns, result, workspace,
                             [&](const mat<real> &rhs, mat<real> &solution) { solve(factor, rhs, solution); });
}

/// @brief Principal inverse block: \f$(A^{-1})_{I I}\f$, in the order of `indices`.
template <sized_factorization F>
inline void inverse_principal_block(const F &factor, view<const idx> indices, mat<real> &result,
                                    inverse_diagonal_workspace &workspace) {
    detail::compute_principal_block(
        static_cast<idx>(factor.size()), indices, result, workspace,
        [&](const mat<real> &rhs, mat<real> &solution) { solve(factor, rhs, solution); });
}

/// @brief Inverse diagonal: \f$\mathrm{diag}(A^{-1})\f$. Allocates.
template <sized_factorization F>
[[nodiscard]] inline vec<real> inverse_diagonal(const F &factor, idx block_size = 64) {
    vec<real> result(static_cast<idx>(factor.size()), 0.0);
    inverse_diagonal_workspace workspace;
    inverse_diagonal(factor, result, workspace, block_size);
    return result;
}

/// @brief Principal inverse block: \f$(A^{-1})_{I I}\f$. Allocates.
template <sized_factorization F>
[[nodiscard]] inline mat<real> inverse_principal_block(const F &factor, view<const idx> indices) {
    mat<real> result;
    inverse_diagonal_workspace workspace;
    inverse_principal_block(factor, indices, result, workspace);
    return result;
}

} // namespace num
