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
using Edge = num::structures::MultiEdge<Float, Index>;

template <typename Float = double, std::integral Index = num::idx>
using Graph = std::vector<std::vector<Edge<Float, Index>>>;

template <typename Float = double, std::integral Index = num::idx>
using Neighbor = num::structures::MultiEdge<Float, Index>;

template <typename Float = double, std::integral Index = num::idx>
struct FactorEntry {
    Index row{};
    Float value{};
};

template <typename Float = double, std::integral Index = num::idx>
struct FactorColumn {
    std::vector<FactorEntry<Float, Index>> entries;
};

template <typename Float = double, std::integral Index = num::idx>
struct CholeskyFactor {
    std::vector<FactorColumn<Float, Index>> columns;
    std::vector<Index> order;
};

template <typename Float = double, std::integral Index = num::idx>
inline Float get_entry(const CholeskyFactor<Float, Index> &F, Index row, Index col) {
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
