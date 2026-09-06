#include "linear/factorization/inverse_diagonal.hpp"
#include "linear/matrix_utils.hpp"
#include <algorithm>
#include <stdexcept>
#include <unordered_map>

namespace num {
namespace {

template <typename Solve>
void compute(idx n, vec &result, inverse_diagonal_workspace &workspace, idx block_size,
             Solve &&solve) {
    if (result.size() != n || block_size == 0) {
        throw std::invalid_argument("inverse_diagonal: invalid result or block size");
    }
    for (idx first = 0; first < n; first += block_size) {
        const idx count = std::min(block_size, n - first);
        workspace.right_hand_sides = identity_columns(n, first, count);
        workspace.solutions = mat(n, count, 0.0);
        solve(workspace.right_hand_sides, workspace.solutions);
        for (idx column = 0; column < count; ++column) {
            result[first + column] = workspace.solutions(first + column, column);
        }
    }
}

template <typename Solve>
void compute_selected(idx n, view<const idx> rows, view<const idx> columns,
                      vec &result, inverse_diagonal_workspace &workspace, Solve &&solve) {
    if (rows.size() != columns.size() || result.size() != rows.size()) {
        throw std::invalid_argument("selected_inverse: request sizes must match");
    }
    array<idx> unique_columns;
    table<idx, idx> position;
    for (idx column : columns) {
        if (column >= n) {
            throw std::out_of_range("selected_inverse: column out of range");
        }
        if (!position.contains(column)) {
            position[column] = unique_columns.size();
            unique_columns.push_back(column);
        }
    }
    workspace.right_hand_sides = mat(n, unique_columns.size(), 0.0);
    for (idx local = 0; local < unique_columns.size(); ++local) {
        workspace.right_hand_sides(unique_columns[local], local) = 1.0;
    }
    workspace.solutions = mat(n, unique_columns.size(), 0.0);
    solve(workspace.right_hand_sides, workspace.solutions);
    for (idx request = 0; request < rows.size(); ++request) {
        if (rows[request] >= n) {
            throw std::out_of_range("selected_inverse: row out of range");
        }
        result[request] = workspace.solutions(rows[request], position[columns[request]]);
    }
}

template <typename Solve>
void compute_principal_block(idx n, view<const idx> indices, mat &result,
                             inverse_diagonal_workspace &workspace, Solve &&solve) {
    if (indices.empty()) {
        workspace.right_hand_sides = mat(n, 0, 0.0);
        workspace.solutions = mat(n, 0, 0.0);
        result = mat(0, 0, 0.0);
        return;
    }
    array<bool> seen(n, false);
    workspace.right_hand_sides = mat(n, indices.size(), 0.0);
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

    workspace.solutions = mat(n, indices.size(), 0.0);
    solve(workspace.right_hand_sides, workspace.solutions);
    result = mat(indices.size(), indices.size(), 0.0);
    for (idx row = 0; row < indices.size(); ++row) {
        for (idx column = 0; column < indices.size(); ++column) {
            result(row, column) = workspace.solutions(indices[row], column);
        }
    }
}

} // namespace

void inverse_diagonal(const lu_result &factor, vec &result, inverse_diagonal_workspace &workspace,
                      idx block_size) {
    compute(factor.LU.rows(), result, workspace, block_size,
            [&](const mat &rhs, mat &solution) { lu_solve(factor, rhs, solution); });
}

void inverse_diagonal(const cholesky_result &factor, vec &result,
                      inverse_diagonal_workspace &workspace, idx block_size) {
    compute(factor.L.rows(), result, workspace, block_size,
            [&](const mat &rhs, mat &solution) { cholesky_solve(factor, rhs, solution); });
}

void inverse_diagonal(const klu_factorization &factor, vec &result, inverse_diagonal_workspace &workspace,
                      idx block_size) {
    compute(factor.size(), result, workspace, block_size,
            [&](const mat &rhs, mat &solution) { factor.solve(rhs, solution); });
}

void inverse_diagonal(const auto_linear_solver &factor, vec &result,
                      inverse_diagonal_workspace &workspace, idx block_size) {
    compute(factor.size(), result, workspace, block_size,
            [&](const mat &rhs, mat &solution) { factor.solve(rhs, solution); });
}

void selected_inverse(const lu_result &factor, view<const idx> rows,
                      view<const idx> columns, vec &result,
                      inverse_diagonal_workspace &workspace) {
    compute_selected(factor.LU.rows(), rows, columns, result, workspace,
                     [&](const mat &rhs, mat &solution) { lu_solve(factor, rhs, solution); });
}

void selected_inverse(const cholesky_result &factor, view<const idx> rows,
                      view<const idx> columns, vec &result,
                      inverse_diagonal_workspace &workspace) {
    compute_selected(
        factor.L.rows(), rows, columns, result, workspace,
        [&](const mat &rhs, mat &solution) { cholesky_solve(factor, rhs, solution); });
}

void selected_inverse(const klu_factorization &factor, view<const idx> rows,
                      view<const idx> columns, vec &result,
                      inverse_diagonal_workspace &workspace) {
    compute_selected(factor.size(), rows, columns, result, workspace,
                     [&](const mat &rhs, mat &solution) { factor.solve(rhs, solution); });
}

void selected_inverse(const auto_linear_solver &factor, view<const idx> rows,
                      view<const idx> columns, vec &result,
                      inverse_diagonal_workspace &workspace) {
    compute_selected(factor.size(), rows, columns, result, workspace,
                     [&](const mat &rhs, mat &solution) { factor.solve(rhs, solution); });
}

void inverse_principal_block(const lu_result &factor, view<const idx> indices, mat &result,
                             inverse_diagonal_workspace &workspace) {
    compute_principal_block(
        factor.LU.rows(), indices, result, workspace,
        [&](const mat &rhs, mat &solution) { lu_solve(factor, rhs, solution); });
}

void inverse_principal_block(const cholesky_result &factor, view<const idx> indices,
                             mat &result, inverse_diagonal_workspace &workspace) {
    compute_principal_block(
        factor.L.rows(), indices, result, workspace,
        [&](const mat &rhs, mat &solution) { cholesky_solve(factor, rhs, solution); });
}

void inverse_principal_block(const klu_factorization &factor, view<const idx> indices, mat &result,
                             inverse_diagonal_workspace &workspace) {
    compute_principal_block(
        factor.size(), indices, result, workspace,
        [&](const mat &rhs, mat &solution) { factor.solve(rhs, solution); });
}

void inverse_principal_block(const auto_linear_solver &factor, view<const idx> indices,
                             mat &result, inverse_diagonal_workspace &workspace) {
    compute_principal_block(
        factor.size(), indices, result, workspace,
        [&](const mat &rhs, mat &solution) { factor.solve(rhs, solution); });
}

} // namespace num
