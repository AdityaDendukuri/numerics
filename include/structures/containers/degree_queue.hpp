/// @file structures/containers/degree_queue.hpp
/// @brief O(1) amortized bucket priority queue for minimum degree elimination in graph algorithms.
#pragma once

#include "core/types.hpp"
#include "structures/debug.hpp"
#include <algorithm>
#include <concepts>
#include <cstdint>
#include <ranges>
#include <stdexcept>
#include <vector>

namespace num::structures {

/// @brief Bucket-based degree priority queue maintaining vertices ordered by integer degree.
/// Provides O(1) insert, erase, rekey, and O(1) amortized pop_min for bounded integer degrees.
template <std::integral Index = num::idx>
class BasicDegreeQueue {
  public:
    using index_type = Index;
    static constexpr Index none = static_cast<Index>(-1);

    /// Construct an empty queue with capacity for n vertices.
    explicit BasicDegreeQueue(Index n = 0) {
        head_.assign(static_cast<std::size_t>(n) + 1, none);
        next_.assign(static_cast<std::size_t>(n), none);
        prev_.assign(static_cast<std::size_t>(n), none);
        degree_.assign(static_cast<std::size_t>(n), 0);
        low_ = 0;
        size_ = 0;
    }

    /// Construct and initialize from a container of degrees.
    template <typename DegreeRange>
    requires (!std::integral<DegreeRange>)
    explicit BasicDegreeQueue(const DegreeRange &degrees) {
        const Index n = static_cast<Index>(degrees.size());
        head_.assign(static_cast<std::size_t>(n) + 1, none);
        next_.assign(static_cast<std::size_t>(n), none);
        prev_.assign(static_cast<std::size_t>(n), none);
        degree_.assign(static_cast<std::size_t>(n), 0);
        low_ = 0;
        size_ = 0;

        for (Index i = 0; i < n; ++i) {
            insert(i, static_cast<Index>(degrees[i]));
        }
    }

    /// Insert vertex v with degree d.
    void insert(Index v, Index d) {
        debug::check_index_bounds(v, static_cast<Index>(degree_.size()), "DegreeQueue::insert");
        if (d >= head_.size()) {
            head_.resize(std::max(static_cast<std::size_t>(d) + 1, 2 * head_.size()), none);
        }
        next_[v] = head_[d];
        prev_[v] = none;
        if (head_[d] != none) {
            prev_[head_[d]] = v;
        }
        head_[d] = v;
        degree_[v] = d;
        if (d < low_) {
            low_ = d;
        }
        ++size_;
    }

    /// Erase vertex v from the queue.
    void erase(Index v) {
        debug::check_index_bounds(v, static_cast<Index>(degree_.size()), "DegreeQueue::erase");
        const Index d = degree_[v];
        if (prev_[v] == none) {
            head_[d] = next_[v];
        } else {
            next_[prev_[v]] = next_[v];
        }
        if (next_[v] != none) {
            prev_[next_[v]] = prev_[v];
        }
        next_[v] = none;
        prev_[v] = none;
        --size_;
    }

    /// Update the degree of vertex v.
    void rekey(Index v, Index d) {
        erase(v);
        insert(v, d);
    }

    /// Pop vertex with minimum degree.
    Index pop_min() {
        if (empty()) {
            throw std::runtime_error("DegreeQueue::pop_min: queue is empty");
        }
        while (low_ < head_.size() && head_[low_] == none) {
            ++low_;
        }
        if (low_ == head_.size()) {
            throw std::runtime_error("DegreeQueue::pop_min: corrupted bucket state");
        }
        const Index v = head_[low_];
        erase(v);
        return v;
    }

    /// Degree of vertex v.
    [[nodiscard]] Index degree_of(Index v) const {
        debug::check_index_bounds(v, static_cast<Index>(degree_.size()), "DegreeQueue::degree_of");
        return degree_[v];
    }

    /// Current minimum degree.
    [[nodiscard]] Index min_degree() const {
        if (empty()) {
            throw std::runtime_error("DegreeQueue::min_degree: queue is empty");
        }
        Index l = low_;
        while (l < head_.size() && head_[l] == none) {
            ++l;
        }
        return l;
    }

    /// Check if queue is empty.
    [[nodiscard]] bool empty() const noexcept { return size_ == 0; }

    /// Number of active elements.
    [[nodiscard]] Index size() const noexcept { return size_; }

    /// Total capacity.
    [[nodiscard]] Index capacity() const noexcept { return static_cast<Index>(degree_.size()); }

    /// Direct access to degree array for elimination updates.
    [[nodiscard]] const std::vector<Index> &degrees() const noexcept { return degree_; }

  private:
    std::vector<Index> head_;
    std::vector<Index> next_;
    std::vector<Index> prev_;
    std::vector<Index> degree_;
    Index low_ = 0;
    Index size_ = 0;
};

using DegreeQueue = BasicDegreeQueue<num::idx>;
using DegreeQueue32 = BasicDegreeQueue<std::uint32_t>;

} // namespace num::structures

namespace num {
using structures::BasicDegreeQueue;
using structures::DegreeQueue;
using structures::DegreeQueue32;
} // namespace num
