/// @file linear/graph/randommat/types.hpp
/// @brief Graph and factor types for randomized linear algebra.
#pragma once

#include "core/types.hpp"
#include "structures/graph/multigraph.hpp"
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace num::randommat {

// Aliases from graph module
template <typename Float = double, std::integral Index = num::idx>
using graph_edge = num::structures::multi_edge<Float, Index>;

/// @brief Weighted multigraph as adjacency lists.
template <typename Float = double, std::integral Index = num::idx>
using adjacency_list = std::vector<std::vector<graph_edge<Float, Index>>>;

template <typename Float = double, std::integral Index = num::idx>
using neighbor = num::structures::multi_edge<Float, Index>;

/// @brief Factor column entry.
template <typename Float = double, std::integral Index = num::idx>
struct factor_entry {
    Index row{};
    Float value{};
};

/// @brief Sparse factor column.
template <typename Float = double, std::integral Index = num::idx>
struct factor_column {
    std::vector<factor_entry<Float, Index>> entries;
};

/// @brief Sparse Cholesky factor by columns, with its elimination order.
template <typename Float = double, std::integral Index = num::idx>
struct cholesky_factor {
    std::vector<factor_column<Float, Index>> columns;
    std::vector<Index> order;
};

/// @brief Factor entry \f$c_{ij}\f$, or 0.
template <typename Float = double, std::integral Index = num::idx>
inline Float get_entry(const cholesky_factor<Float, Index> &F, Index row, Index col) {
    if (col >= F.columns.size()) {
        return static_cast<Float>(0);
    }
    for (const auto &entry : F.columns[col].entries) {
        if (entry.row == row) {
            return entry.value;
        }
    }
    return static_cast<Float>(0);
}

} // namespace num::randommat
