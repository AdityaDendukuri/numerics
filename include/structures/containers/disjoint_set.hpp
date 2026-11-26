/// @file structures/containers/disjoint_set.hpp
/// @brief Union-find with path compression and union by rank.
#pragma once

#include "core/types.hpp"
#include "structures/concepts.hpp"
#include "structures/debug.hpp"
#include <concepts>
#include <numeric>
#include <vector>

namespace num {

/// @brief Disjoint sets stored as a forest: `find` and `unite` in nearly \f$O(1)\f$ amortized time.
template <std::integral Index = num::idx>
class basic_disjoint_set {
  public:
    using index_type = Index;

    /// @brief Construct `n` sets of one element each: \f$\{0\}, \ldots, \{n-1\}\f$.
    explicit basic_disjoint_set(Index n = 0)
        : parent_(n), rank_(n, 0), size_(n, 1), count_(n) {
        std::iota(parent_.begin(), parent_.end(), static_cast<Index>(0));
    }

    /// @brief Return the representative of \f$[u]\f$, shortening the path.
    Index find(Index u) {
        structures::debug::check_index_bounds(u, static_cast<Index>(parent_.size()), "disjoint_set::find");
        Index root = u;
        while (root != parent_[root]) {
            root = parent_[root];
        }
        // Two-pass path compression
        Index curr = u;
        while (curr != root) {
            Index nxt = parent_[curr];
            parent_[curr] = root;
            curr = nxt;
        }
        return root;
    }

    /// @brief Return the representative of \f$[u]\f$ without changing the forest.
    [[nodiscard]] Index find(Index u) const {
        structures::debug::check_index_bounds(u, static_cast<Index>(parent_.size()), "disjoint_set::find");
        while (u != parent_[u]) {
            u = parent_[u];
        }
        return u;
    }

    /// @brief Merge \f$[u]\f$ and \f$[v]\f$; return whether they were different sets.
    bool unite(Index u, Index v) {
        structures::debug::check_index_bounds(u, static_cast<Index>(parent_.size()), "disjoint_set::unite u");
        structures::debug::check_index_bounds(v, static_cast<Index>(parent_.size()), "disjoint_set::unite v");

        Index root_u = find(u);
        Index root_v = find(v);

        if (root_u == root_v) {
            return false;
        }

        // Union by rank
        if (rank_[root_u] < rank_[root_v]) {
            parent_[root_u] = root_v;
            size_[root_v] += size_[root_u];
        } else {
            parent_[root_v] = root_u;
            size_[root_u] += size_[root_v];
            if (rank_[root_u] == rank_[root_v]) {
                rank_[root_u]++;
            }
        }
        count_--;
        return true;
    }

    /// @brief Test whether \f$u \sim v\f$.
    [[nodiscard]] bool connected(Index u, Index v) {
        return find(u) == find(v);
    }

    /// @brief Return the size of \f$[u]\f$.
    [[nodiscard]] Index component_size(Index u) {
        return size_[find(u)];
    }

    /// @brief Return the number of sets.
    [[nodiscard]] Index count() const noexcept {
        return count_;
    }

    /// @brief Return the number of elements.
    [[nodiscard]] Index size() const noexcept {
        return static_cast<Index>(parent_.size());
    }

    /// @brief Reset to `n` sets of one element each.
    void reset(Index n = 0) {
        if (n == 0) n = static_cast<Index>(parent_.size());
        parent_.resize(n);
        std::iota(parent_.begin(), parent_.end(), static_cast<Index>(0));
        rank_.assign(n, 0);
        size_.assign(n, 1);
        count_ = n;
    }

    /// @brief Return all sets, as lists of elements.
    [[nodiscard]] array<array<Index>> components() {
        array<array<Index>> comp_map(parent_.size());
        for (Index i = 0; i < static_cast<Index>(parent_.size()); ++i) {
            comp_map[find(i)].push_back(i);
        }
        array<array<Index>> result;
        result.reserve(count_);
        for (auto &c : comp_map) {
            if (!c.empty()) {
                result.push_back(std::move(c));
            }
        }
        return result;
    }

  private:
    array<Index> parent_;
    array<Index> rank_;
    array<Index> size_;
    Index count_ = 0;
};

/// @brief Disjoint set with `idx` indices.
using disjoint_set = basic_disjoint_set<num::idx>;

/// @brief Disjoint set with 32-bit indices.
using disjoint_set_32 = basic_disjoint_set<uint32_t>;

static_assert(concepts::equivalence_relation<disjoint_set, num::idx>,
              "disjoint_set must satisfy equivalence_relation concept");
static_assert(concepts::equivalence_relation<disjoint_set_32, uint32_t>,
              "disjoint_set_32 must satisfy equivalence_relation concept");

static_assert(equivalence_relation<disjoint_set>);
} // namespace num
