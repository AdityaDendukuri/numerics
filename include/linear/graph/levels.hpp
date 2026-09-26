/// @file linear/graph/levels.hpp
/// @brief Block-tridiagonal level labels from graph distance.
#pragma once

#include "container/vector.hpp"
#include "core/types.hpp"
#include "linear/sparse/sparse.hpp"
#include <queue>
#include <stdexcept>

namespace num {

/// @brief Label each row by its graph distance from a seed set.
///
/// The undirected support graph joins two distinct rows whenever either
/// corresponding off-diagonal entry is nonzero. Grouping rows by their distance
/// from a seed set and ordering the groups by increasing distance makes the
/// matrix block-tridiagonal, because an edge can only join rows whose distances
/// differ by at most one. Feeding the result to `build_block_order` therefore
/// yields a layout `factor_block_lu` accepts.
///
/// Rows in a component no seed reaches are labelled from their own component's
/// breadth-first search, restarting at distance zero. That is sound for the same
/// reason: no edge crosses between components, so no block picks up a coupling
/// beyond its neighbours. A caller that needs one global ordering should seed
/// every component, as Proposition 4.2 assumes.
///
/// The number of distinct levels controls the block width, and so the cost of
/// the factorization: with L balanced levels the elimination work falls from
/// \f$O(n^3)\f$ to \f$O(n^3/L^2)\f$. Seeding a single row of a long chain gives
/// the most levels; seeding many rows at once gives fewer and wider ones.
///
/// @param A Square CSR matrix whose support graph is traversed.
/// @param seeds Rows at distance zero. Empty means every component starts from
///        its lowest-numbered row.
/// @return One level label per row.
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

/// @brief The same labels from a single seed row.
[[nodiscard]] inline array<idx> graph_distance_levels(const spmat &A, idx seed) {
    const array<idx> seeds{seed};
    return graph_distance_levels(A, seeds);
}

} // namespace num
