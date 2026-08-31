/// @file multi_index.hpp
/// @brief MultiIndex: fixed-capacity stack-allocated integer tuple for discrete state
/// spaces.
#pragma once

#include "core/types.hpp"
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <iostream>

namespace num {

/// @brief Stack-allocated fixed-capacity integer multi-index for discrete state spaces.
struct MultiIndex {
    static constexpr std::size_t k_max_dim = 8;

    int8_t n_ = 0;
    int8_t _pad[3] = {}; // align data_ to 4 bytes
    int data_[k_max_dim] = {};

    MultiIndex() = default;

    /// Construct from up to `k_max_dim` coordinate values.
    MultiIndex(std::initializer_list<int> il) : n_(static_cast<int8_t>(il.size())) {
        assert(il.size() <= k_max_dim);
        std::size_t i = 0;
        for (int v : il) {
            data_[i++] = v;
        }
    }

    /// Construct an n-dimensional index with every coordinate set to val.
    explicit MultiIndex(std::size_t n, int val = 0) : n_(static_cast<int8_t>(n)) {
        assert(n <= k_max_dim);
        for (std::size_t i = 0; i < n; ++i) {
            data_[i] = val;
        }
    }

    /// Return the active coordinate count.
    [[nodiscard]] std::size_t size() const noexcept { return static_cast<std::size_t>(n_); }

    int &operator[](std::size_t i) noexcept { return data_[i]; }
    int operator[](std::size_t i) const noexcept { return data_[i]; }

    int *begin() noexcept { return data_; }
    int *end() noexcept { return data_ + n_; }
    [[nodiscard]] const int *begin() const noexcept { return data_; }
    [[nodiscard]] const int *end() const noexcept { return data_ + n_; }

    bool operator==(const MultiIndex &o) const noexcept {
        if (n_ != o.n_) {
            return false;
        }
        for (int8_t i = 0; i < n_; ++i) {
            if (data_[i] != o.data_[i]) {
                return false;
            }
        }
        return true;
    }
    bool operator!=(const MultiIndex &o) const noexcept { return !(*this == o); }
    bool operator<(const MultiIndex &o) const noexcept {
        if (n_ != o.n_) {
            return n_ < o.n_;
        }
        for (int8_t i = 0; i < n_; ++i) {
            if (data_[i] != o.data_[i]) {
                return data_[i] < o.data_[i];
            }
        }
        return false;
    }

    /// Add corresponding coordinates of equal-dimensional indices.
    MultiIndex operator+(const MultiIndex &o) const {
        MultiIndex r(size());
        for (int8_t i = 0; i < n_; ++i) {
            r.data_[i] = data_[i] + o.data_[i];
        }
        return r;
    }

    /// Subtract corresponding coordinates of equal-dimensional indices.
    MultiIndex operator-(const MultiIndex &o) const {
        MultiIndex r(size());
        for (int8_t i = 0; i < n_; ++i) {
            r.data_[i] = data_[i] - o.data_[i];
        }
        return r;
    }
};

} // namespace num

// --- std::hash specialization for num::MultiIndex using Boost hash_combine ---
namespace std {
template <>
struct hash<num::MultiIndex> {
    std::size_t operator()(const num::MultiIndex &x) const noexcept {
        std::size_t seed = x.size();
        for (int8_t i = 0; i < x.n_; ++i) {
            seed ^= static_cast<std::size_t>(x.data_[i]) + 0x9e3779b9u + (seed << 6) + (seed >> 2);
        }
        return seed;
    }
};
} // namespace std
