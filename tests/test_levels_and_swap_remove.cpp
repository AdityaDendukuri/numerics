/// @file tests/test_levels_and_swap_remove.cpp
/// @brief Graph-distance block labels and index-preserving unordered removal.
///
/// Both utilities exist to make a downstream invariant hold, so the tests assert
/// the invariant rather than the implementation: levels must make the matrix
/// block-tridiagonal, and a swap-remove must leave the surviving elements
/// reachable through the reported mapping.

#include "container/swap_remove.hpp"
#include "linear/factorization/block_tridiagonal.hpp"
#include "linear/graph/levels.hpp"
#include "linear/sparse/sparse.hpp"
#include <gtest/gtest.h>
#include <algorithm>
#include <set>

using namespace num;

namespace {

/// A path graph, whose distances from one end are 0, 1, 2, ... in order.
spmat path_graph(idx n) {
    array<idx> rows, columns;
    array<real> values;
    for (idx i = 0; i < n; ++i) {
        rows.push_back(i);
        columns.push_back(i);
        values.push_back(2.0);
        if (i + 1 < n) {
            rows.push_back(i);
            columns.push_back(i + 1);
            values.push_back(-1.0);
            rows.push_back(i + 1);
            columns.push_back(i);
            values.push_back(-1.0);
        }
    }
    return spmat::from_triplets(n, n, rows, columns, values);
}

/// A `width` by `height` grid, whose distances from a corner are the L1 distances.
spmat grid_graph(idx width, idx height) {
    const idx n = width * height;
    array<idx> rows, columns;
    array<real> values;
    const auto at = [&](idx x, idx y) { return (y * width) + x; };
    for (idx y = 0; y < height; ++y) {
        for (idx x = 0; x < width; ++x) {
            rows.push_back(at(x, y));
            columns.push_back(at(x, y));
            values.push_back(4.0);
            if (x + 1 < width) {
                rows.push_back(at(x, y));
                columns.push_back(at(x + 1, y));
                values.push_back(-1.0);
                rows.push_back(at(x + 1, y));
                columns.push_back(at(x, y));
                values.push_back(-1.0);
            }
            if (y + 1 < height) {
                rows.push_back(at(x, y));
                columns.push_back(at(x, y + 1));
                values.push_back(-1.0);
                rows.push_back(at(x, y + 1));
                columns.push_back(at(x, y));
                values.push_back(-1.0);
            }
        }
    }
    return spmat::from_triplets(n, n, rows, columns, values);
}

/// True when no stored entry joins rows whose levels differ by more than one.
bool is_block_tridiagonal(const spmat &A, view<const idx> levels) {
    for (idx row = 0; row < A.n_rows(); ++row) {
        for (idx entry = A.row_ptr()[row]; entry < A.row_ptr()[row + 1]; ++entry) {
            if (A.values()[entry] == 0.0) {
                continue;
            }
            const idx column = A.col_idx()[entry];
            const idx high = std::max(levels[row], levels[column]);
            const idx low = std::min(levels[row], levels[column]);
            if (high - low > 1) {
                return false;
            }
        }
    }
    return true;
}

} // namespace

TEST(GraphDistanceLevels, LabelsAPathByItsDistanceFromAnEnd) {
    constexpr idx n = 8;
    const array<idx> levels = graph_distance_levels(path_graph(n), idx{0});
    ASSERT_EQ(levels.size(), n);
    for (idx i = 0; i < n; ++i) {
        EXPECT_EQ(levels[i], i) << "row " << i;
    }
}

TEST(GraphDistanceLevels, MakesAGridBlockTridiagonal) {
    const spmat A = grid_graph(5, 4);
    const array<idx> levels = graph_distance_levels(A, idx{0});
    EXPECT_TRUE(is_block_tridiagonal(A, levels));

    // The corner of a 5 by 4 grid is at L1 distance 0 through 7.
    const std::set<idx> distinct(levels.begin(), levels.end());
    EXPECT_EQ(distinct.size(), 8U);
}

TEST(GraphDistanceLevels, SeedingSeveralRowsGivesFewerWiderBlocks) {
    const spmat A = path_graph(20);
    const array<idx> one = graph_distance_levels(A, idx{0});
    const array<idx> several{0, 10};
    const array<idx> many = graph_distance_levels(A, several);

    const std::set<idx> from_one(one.begin(), one.end());
    const std::set<idx> from_many(many.begin(), many.end());
    EXPECT_LT(from_many.size(), from_one.size());
    EXPECT_TRUE(is_block_tridiagonal(A, many));
}

TEST(GraphDistanceLevels, LabelsAnUnreachedComponentFromItsOwnStart) {
    // Two disjoint paths, seeded only in the first. No edge crosses between
    // them, so restarting the second at zero still leaves the matrix
    // block-tridiagonal.
    array<idx> rows, columns;
    array<real> values;
    const auto edge = [&](idx i, idx j) {
        rows.push_back(i);
        columns.push_back(j);
        values.push_back(-1.0);
        rows.push_back(j);
        columns.push_back(i);
        values.push_back(-1.0);
    };
    for (idx i = 0; i < 6; ++i) {
        rows.push_back(i);
        columns.push_back(i);
        values.push_back(2.0);
    }
    edge(0, 1);
    edge(1, 2);
    edge(3, 4);
    edge(4, 5);
    const spmat A = spmat::from_triplets(6, 6, rows, columns, values);

    const array<idx> levels = graph_distance_levels(A, idx{0});
    EXPECT_EQ(levels[0], 0);
    EXPECT_EQ(levels[1], 1);
    EXPECT_EQ(levels[2], 2);
    EXPECT_EQ(levels[3], 0);
    EXPECT_EQ(levels[4], 1);
    EXPECT_EQ(levels[5], 2);
    EXPECT_TRUE(is_block_tridiagonal(A, levels));
}

TEST(GraphDistanceLevels, FeedsBuildBlockOrderDirectly) {
    const spmat A = grid_graph(4, 4);
    const array<idx> levels = graph_distance_levels(A, idx{0});
    const detail::block_layout layout = detail::build_block_order(levels);
    EXPECT_EQ(layout.offsets.back(), A.n_rows());
    EXPECT_NO_THROW(detail::validate_block_structure(A, layout));
}

TEST(GraphDistanceLevels, RejectsASeedOutsideTheMatrix) {
    const spmat A = path_graph(4);
    const array<idx> seeds{9};
    EXPECT_THROW((void)graph_distance_levels(A, seeds), std::out_of_range);
}

TEST(SwapRemove, FillsHolesFromTheTail) {
    array<int> values{10, 11, 12, 13, 14, 15};
    const array<idx> removed{1, 3};
    swap_remove(values, removed);

    // Holes 1 and 3 take the survivors 4 and 5, in that order.
    ASSERT_EQ(values.size(), 4U);
    EXPECT_EQ(values[0], 10);
    EXPECT_EQ(values[1], 14);
    EXPECT_EQ(values[2], 12);
    EXPECT_EQ(values[3], 15);
}

TEST(SwapRemove, IsAPlainTruncationWhenTheTailIsRemoved) {
    array<int> values{1, 2, 3, 4, 5};
    const array<idx> removed{3, 4};
    EXPECT_TRUE(swap_remove_plan(values.size(), removed).empty());
    swap_remove(values, removed);
    ASSERT_EQ(values.size(), 3U);
    EXPECT_EQ(values[2], 3);
}

TEST(SwapRemove, RemoveIndicesKeepsTheOrderOfTheSurvivors) {
    array<int> values{10, 11, 12, 13, 14, 15};
    const array<idx> removed{3, 1};
    remove_indices(values, removed);
    EXPECT_EQ(values, (array<int>{10, 12, 14, 15}));
    const array<idx> outside{6};
    EXPECT_THROW(remove_indices(values, outside), std::out_of_range);
}

TEST(SwapRemove, DoesNotDependOnTheOrderOfTheRemovedIndices) {
    const array<idx> ascending{1, 4};
    const array<idx> descending{4, 1};
    const array<index_move> first = swap_remove_plan(7, ascending);
    const array<index_move> second = swap_remove_plan(7, descending);
    ASSERT_EQ(first.size(), second.size());
    for (idx k = 0; k < first.size(); ++k) {
        EXPECT_EQ(first[k].from, second[k].from);
        EXPECT_EQ(first[k].to, second[k].to);
    }
}

TEST(SwapRemove, KeepsParallelArraysConsistent) {
    array<int> keys{0, 1, 2, 3, 4, 5, 6};
    array<double> weights{0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    const array<idx> removed{0, 2, 5};
    swap_remove(keys, removed);
    swap_remove(weights, removed);

    ASSERT_EQ(keys.size(), weights.size());
    for (idx i = 0; i < keys.size(); ++i) {
        EXPECT_DOUBLE_EQ(weights[i], static_cast<double>(keys[i]))
            << "arrays disagreed at " << i;
    }
}

TEST(SwapRemoveMap, ReportsTheNewIndexOfEverySurvivor) {
    array<int> values{0, 1, 2, 3, 4, 5};
    const array<idx> removed{1, 3};
    const array<idx> mapping = swap_remove_map(values.size(), removed);
    swap_remove(values, removed);

    EXPECT_EQ(mapping[1], removed_index);
    EXPECT_EQ(mapping[3], removed_index);
    for (idx old = 0; old < mapping.size(); ++old) {
        if (mapping[old] != removed_index) {
            EXPECT_EQ(values[mapping[old]], static_cast<int>(old))
                << "survivor " << old << " was not where the map said";
        }
    }
}

TEST(SwapRemoveMap, IsTheIdentityWhenNothingIsRemoved) {
    const array<idx> mapping = swap_remove_map(5, {});
    for (idx i = 0; i < 5; ++i) {
        EXPECT_EQ(mapping[i], i);
    }
}

TEST(SwapRemove, RejectsAnIndexOutsideTheRange) {
    const array<idx> removed{7};
    EXPECT_THROW((void)swap_remove_plan(4, removed), std::out_of_range);
}
