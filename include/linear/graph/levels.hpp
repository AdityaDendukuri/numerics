/// @file linear/graph/levels.hpp
/// @brief Block-tridiagonal levels from graph distance.
#pragma once

#include "container/vector.hpp"
#include "core/types.hpp"
#include "linear/sparse/sparse.hpp"
#include <queue>
#include <stdexcept>

namespace num {

/// @brief Label each row by its graph distance from the seeds: \f$\ell_i = d(i, \mathrm{seeds})\f$.
/// Grouping by level makes \f$A\f$ block tridiagonal.
/// Empty `seeds` starts each component at its lowest row.
/// @throws std::invalid_argument If `A` is not square or a seed is out of range.
[[nodiscard]] inline array<idx> graph_distance_levels(const spmat &A, view<const idx> seeds) {
    const idx n = A.n_rows();
    if (A.n_cols() != n) {
        throw std::invalid_argument("graph_distance_levels: matrix must be square");
    }

    // The undirected support graph, as adjacency lists. An entry counts as an
    // edge in both directions, which is what makes the level bound symmetric.
    array<array<idx>> neighbours(n);
    for (idx row = 0; row < n; ++row) {
        for (idx entry = A.row_ptr()[row]; entry < A.row_ptr()[row + 1]; ++entry) {
            const idx column = A.col_idx()[entry];
            if (column != row && A.values()[entry] != 0.0) {
                neighbours[row].push_back(column);
                neighbours[column].push_back(row);
            }
        }
    }

    constexpr idx unvisited = static_cast<idx>(-1);
    array<idx> level(n, unvisited);
    std::queue<idx> frontier;
    for (idx seed : seeds) {
        if (seed >= n) {
            throw std::out_of_range("graph_distance_levels: seed is outside the matrix");
        }
        if (level[seed] == unvisited) {
            level[seed] = 0;
            frontier.push(seed);
        }
    }

    const auto traverse = [&] {
        while (!frontier.empty()) {
            const idx row = frontier.front();
            frontier.pop();
            for (idx neighbour : neighbours[row]) {
                if (level[neighbour] == unvisited) {
                    level[neighbour] = level[row] + 1;
                    frontier.push(neighbour);
                }
            }
        }
    };
    traverse();

    for (idx row = 0; row < n; ++row) {
        if (level[row] == unvisited) {
            level[row] = 0;
            frontier.push(row);
            traverse();
        }
    }
    return level;
}

/// @brief Label each row by its graph distance from one seed: \f$\ell_i = d(i, s)\f$.
[[nodiscard]] inline array<idx> graph_distance_levels(const spmat &A, idx seed) {
    const array<idx> seeds{seed};
    return graph_distance_levels(A, seeds);
}

} // namespace num
