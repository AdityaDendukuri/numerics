/// @file structures/containers/indexed_priority_queue.hpp
/// @brief Indexed binary heap.
#pragma once

#include "core/types.hpp"
#include "structures/concepts.hpp"
#include "structures/debug.hpp"
#include <concepts>
#include <functional>
#include <utility>
#include <vector>

namespace num {

/// @brief Indexed priority queue: items in `[0, capacity)`, \f$O(\log n)\f$ push, pop and update.
/// `comp(a, b)` true means `a` has priority; the default is a min-heap.
template <typename Key = double, std::integral Index = num::idx, typename Compare = std::less<Key>>
class indexed_priority_queue {
  public:
    using key_type = Key;
    using index_type = Index;
    using compare_type = Compare;

    /// @brief Empty queue for indices below `capacity`.
    explicit indexed_priority_queue(Index capacity = 0, Compare comp = Compare{})
        : comp_(comp), capacity_(capacity), size_(0),
          heap_(capacity + 1, 0), pos_(capacity, 0), keys_(capacity) {}

    /// @brief Insert `index` with `key`.
    void push(Index index, const Key &key) {
        structures::debug::check_index_bounds(index, capacity_, "indexed_priority_queue::push");
        structures::debug::check_not_contains(contains(index), index, "indexed_priority_queue::push");

        size_++;
        heap_[size_] = index;
        pos_[index] = size_;
        keys_[index] = key;
        swim(size_);
    }

    /// @brief Remove the top element.
    void pop() {
        if (empty()) {
            structures::debug::check_index_bounds(static_cast<Index>(0), static_cast<Index>(0),
                                          "indexed_priority_queue::pop empty queue");
            return;
        }
        Index top_idx = heap_[1];
        swap_nodes(1, size_);
        pos_[top_idx] = 0;
        size_--;
        sink(1);
    }

    /// @brief Index of the top element.
    [[nodiscard]] Index top_index() const {
        if (empty()) {
            structures::debug::check_index_bounds(static_cast<Index>(0), static_cast<Index>(0),
                                          "indexed_priority_queue::top_index empty queue");
        }
        return heap_[1];
    }

    /// @brief Key of the top element.
    [[nodiscard]] const Key &top_key() const {
        if (empty()) {
            structures::debug::check_index_bounds(static_cast<Index>(0), static_cast<Index>(0),
                                          "indexed_priority_queue::top_key empty queue");
        }
        return keys_[heap_[1]];
    }

    /// @brief Whether `index` is queued.
    [[nodiscard]] bool contains(Index index) const noexcept {
        if (index >= capacity_) return false;
        return pos_[index] != 0;
    }

    /// @brief Key of `index`.
    [[nodiscard]] const Key &key_of(Index index) const {
        structures::debug::check_index_bounds(index, capacity_, "indexed_priority_queue::key_of");
        structures::debug::check_contains(contains(index), index, "indexed_priority_queue::key_of");
        return keys_[index];
    }

    /// @brief Set the key of `index`.
    void update(Index index, const Key &new_key) {
        structures::debug::check_index_bounds(index, capacity_, "indexed_priority_queue::update");
        if (!contains(index)) {
            push(index, new_key);
            return;
        }
        keys_[index] = new_key;
        swim(pos_[index]);
        sink(pos_[index]);
    }

    /// @brief Set the key of `index` to a higher-priority value.
    void improve_key(Index index, const Key &new_key) {
        structures::debug::check_index_bounds(index, capacity_, "indexed_priority_queue::improve_key");
        structures::debug::check_contains(contains(index), index, "indexed_priority_queue::improve_key");
        if (comp_(new_key, keys_[index])) {
            keys_[index] = new_key;
            swim(pos_[index]);
        }
    }

    /// @brief Erase `index`.
    void erase(Index index) {
        structures::debug::check_index_bounds(index, capacity_, "indexed_priority_queue::erase");
        structures::debug::check_contains(contains(index), index, "indexed_priority_queue::erase");
        Index heap_idx = pos_[index];
        swap_nodes(heap_idx, size_);
        pos_[index] = 0;
        size_--;
        if (heap_idx <= size_) {
            swim(heap_idx);
            sink(heap_idx);
        }
    }

    /// @brief Whether the queue is empty.
    [[nodiscard]] bool empty() const noexcept { return size_ == 0; }

    /// @brief Number of queued items.
    [[nodiscard]] Index size() const noexcept { return size_; }

    /// @brief Index capacity.
    [[nodiscard]] Index capacity() const noexcept { return capacity_; }

    /// @brief Remove every item.
    void clear() noexcept {
        for (Index i = 1; i <= size_; ++i) {
            pos_[heap_[i]] = 0;
        }
        size_ = 0;
    }

  private:
    void swap_nodes(Index i, Index j) {
        std::swap(heap_[i], heap_[j]);
        pos_[heap_[i]] = i;
        pos_[heap_[j]] = j;
    }

    void swim(Index k) {
        while (k > 1 && comp_(keys_[heap_[k]], keys_[heap_[k / 2]])) {
            swap_nodes(k, k / 2);
            k = k / 2;
        }
    }

    void sink(Index k) {
        while (2 * k <= size_) {
            Index j = 2 * k;
            if (j < size_ && comp_(keys_[heap_[j + 1]], keys_[heap_[j]])) {
                j++;
            }
            if (!comp_(keys_[heap_[j]], keys_[heap_[k]])) {
                break;
            }
            swap_nodes(k, j);
            k = j;
        }
    }

    Compare comp_;
    Index capacity_ = 0;
    Index size_ = 0;
    array<Index> heap_;
    array<Index> pos_;
    array<Key> keys_;
};

/// @brief Indexed min-heap.
template <typename Key = double, std::integral Index = num::idx>
using min_indexed_pq = indexed_priority_queue<Key, Index, std::less<Key>>;

/// @brief Indexed max-heap.
template <typename Key = double, std::integral Index = num::idx>
using max_indexed_pq = indexed_priority_queue<Key, Index, std::greater<Key>>;

static_assert(concepts::addressable_priority_queue<min_indexed_pq<double, num::idx>, double, num::idx>,
              "min_indexed_pq must satisfy addressable_priority_queue");

static_assert(addressable_priority_queue<min_indexed_pq<real, idx>, real>);
} // namespace num
