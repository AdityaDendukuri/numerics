/// @file structures/containers/degree_queue.hpp
/// @brief Bucket priority queue by integer degree.
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

/// @brief Degree queue: vertices bucketed by degree, \f$O(1)\f$ updates and amortized `pop_min`.
template <std::integral Index = num::idx>
class basic_degree_queue {
  public:
    using index_type = Index;
    static constexpr Index none = static_cast<Index>(-1);

    /// @brief Empty queue with capacity `n`.
    explicit basic_degree_queue(Index n = 0) {
        head_.assign(static_cast<std::size_t>(n) + 1, none);
        next_.assign(static_cast<std::size_t>(n), none);
        prev_.assign(static_cast<std::size_t>(n), none);
        degree_.assign(static_cast<std::size_t>(n), 0);
        low_ = 0;
        size_ = 0;
    }

    /// @brief Queue initialized from a range of degrees.
    template <typename DegreeRange>
    requires (!std::integral<DegreeRange>)
    explicit basic_degree_queue(const DegreeRange &degrees) {
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

    /// @brief Insert vertex `v` with degree `d`.
    void insert(Index v, Index d) {
        debug::check_index_bounds(v, static_cast<Index>(degree_.size()), "degree_queue::insert");
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

    /// @brief Erase vertex `v`.
    void erase(Index v) {
        debug::check_index_bounds(v, static_cast<Index>(degree_.size()), "degree_queue::erase");
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

    /// @brief Set the degree of `v` to `d`.
    void rekey(Index v, Index d) {
        erase(v);
        insert(v, d);
    }

    /// @brief Pop a vertex of minimum degree.
    Index pop_min() {
        if (empty()) {
            throw std::runtime_error("degree_queue::pop_min: queue is empty");
        }
        while (low_ < head_.size() && head_[low_] == none) {
            ++low_;
        }
        if (low_ == head_.size()) {
            throw std::runtime_error("degree_queue::pop_min: corrupted bucket state");
        }
        const Index v = head_[low_];
        erase(v);
        return v;
    }

    /// @brief Degree of `v`.
    [[nodiscard]] Index degree_of(Index v) const {
        debug::check_index_bounds(v, static_cast<Index>(degree_.size()), "degree_queue::degree_of");
        return degree_[v];
    }

    /// @brief Minimum degree.
    [[nodiscard]] Index min_degree() const {
        if (empty()) {
            throw std::runtime_error("degree_queue::min_degree: queue is empty");
        }
        Index l = low_;
        while (l < head_.size() && head_[l] == none) {
            ++l;
        }
        return l;
    }

    /// @brief Whether the queue is empty.
    [[nodiscard]] bool empty() const noexcept { return size_ == 0; }

    /// @brief Number of queued vertices.
    [[nodiscard]] Index size() const noexcept { return size_; }

    /// @brief Capacity.
    [[nodiscard]] Index capacity() const noexcept { return static_cast<Index>(degree_.size()); }

    /// @brief Degree array.
    [[nodiscard]] const array<Index> &degrees() const noexcept { return degree_; }

  private:
    array<Index> head_;
    array<Index> next_;
    array<Index> prev_;
    array<Index> degree_;
    Index low_ = 0;
    Index size_ = 0;
};

using degree_queue = basic_degree_queue<num::idx>;
using degree_queue_32 = basic_degree_queue<std::uint32_t>;

} // namespace num::structures

namespace num {
using structures::basic_degree_queue;
using structures::degree_queue;
using structures::degree_queue_32;
} // namespace num
