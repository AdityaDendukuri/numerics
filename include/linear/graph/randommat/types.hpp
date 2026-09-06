/// @file linear/graph/randommat/types.hpp
/// @brief Core types and factor representations for Randomized Numerical Linear Algebra (RandNLA).
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

template <typename Float = double, std::integral Index = num::idx>
using graph = std::vector<std::vector<graph_edge<Float, Index>>>;

template <typename Float = double, std::integral Index = num::idx>
using neighbor = num::structures::multi_edge<Float, Index>;

template <typename Float = double, std::integral Index = num::idx>
struct factor_entry {
    Index row{};
    Float value{};
};

template <typename Float = double, std::integral Index = num::idx>
struct factor_column {
    std::vector<factor_entry<Float, Index>> entries;
};

template <typename Float = double, std::integral Index = num::idx>
struct cholesky_factor {
    std::vector<factor_column<Float, Index>> columns;
    std::vector<Index> order;
};

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
