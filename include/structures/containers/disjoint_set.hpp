/// @file structures/containers/disjoint_set.hpp
/// @brief High-performance Disjoint-Set (Union-Find) with path compression and union by rank.
#pragma once

#include "core/types.hpp"
#include "structures/concepts.hpp"
#include "structures/debug.hpp"
#include <concepts>
#include <numeric>
#include <vector>

namespace num {

/// @brief Disjoint-Set forest data structure supporting near O(1) amortized operations.
/// @tparam Index Integer type used for vertex/element indices (e.g. num::idx, uint32_t, int).
template <std::integral Index = num::idx>
class basic_disjoint_set {
  public:
    using index_type = Index;

    /// Construct a disjoint-set structure with n singleton sets: {0}, {1}, ..., {n-1}.
    explicit basic_disjoint_set(Index n = 0)
        : parent_(n), rank_(n, 0), size_(n, 1), count_(n) {
        std::iota(parent_.begin(), parent_.end(), static_cast<Index>(0));
    }

    /// Find the representative root of the set containing element u with path compression.
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

    /// Find representative root (const view without mutation).
    [[nodiscard]] Index find(Index u) const {
        structures::debug::check_index_bounds(u, static_cast<Index>(parent_.size()), "disjoint_set::find");
        while (u != parent_[u]) {
            u = parent_[u];
        }
        return u;
    }

    /// Unite the sets containing elements u and v.
    /// @return true if u and v were in different sets, false if already in the same set.
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

    /// Check whether elements u and v belong to the same connected component.
    [[nodiscard]] bool connected(Index u, Index v) {
        return find(u) == find(v);
    }

    /// Return the number of elements in the component containing element u.
    [[nodiscard]] Index component_size(Index u) {
        return size_[find(u)];
    }

    /// Return the total number of disjoint components remaining.
    [[nodiscard]] Index count() const noexcept {
        return count_;
    }

    /// Return the total number of elements.
    [[nodiscard]] Index size() const noexcept {
        return static_cast<Index>(parent_.size());
    }

    /// Reset the structure back to n singleton sets.
    void reset(Index n = 0) {
        if (n == 0) n = static_cast<Index>(parent_.size());
        parent_.resize(n);
        std::iota(parent_.begin(), parent_.end(), static_cast<Index>(0));
        rank_.assign(n, 0);
        size_.assign(n, 1);
        count_ = n;
    }

    /// Return all connected components as lists of element indices.
    [[nodiscard]] std::vector<std::vector<Index>> components() {
        std::vector<std::vector<Index>> comp_map(parent_.size());
        for (Index i = 0; i < static_cast<Index>(parent_.size()); ++i) {
            comp_map[find(i)].push_back(i);
        }
        std::vector<std::vector<Index>> result;
        result.reserve(count_);
        for (auto &c : comp_map) {
            if (!c.empty()) {
                result.push_back(std::move(c));
            }
        }
        return result;
    }

  private:
    std::vector<Index> parent_;
    std::vector<Index> rank_;
    std::vector<Index> size_;
    Index count_ = 0;
};

/// Canonical 64-bit disjoint_set alias
using disjoint_set = basic_disjoint_set<num::idx>;

/// Compact 32-bit disjoint_set alias (halves memory overhead)
using disjoint_set_32 = basic_disjoint_set<uint32_t>;

static_assert(concepts::equivalence_relation<disjoint_set, num::idx>,
              "disjoint_set must satisfy equivalence_relation concept");
static_assert(concepts::equivalence_relation<disjoint_set_32, uint32_t>,
              "disjoint_set_32 must satisfy equivalence_relation concept");

} // namespace num
