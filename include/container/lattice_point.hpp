/// @file container/lattice_point.hpp
/// @brief A point of the integer lattice, stored inline.
#pragma once

#include <compare>
#include <cstddef>
#include <functional>

namespace num {

/// @brief A point of \f$\mathbb{Z}^D\f$: `D` ints stored inline, with lattice addition,
/// lexicographic order and a hash, so it can key a map.
template <int D>
struct lattice_point {
    static_assert(D > 0, "lattice_point needs at least one coordinate");

    int coords[D] = {};

    [[nodiscard]] static constexpr std::size_t size() noexcept { return D; }
    constexpr int &operator[](std::size_t i) noexcept { return coords[i]; }
    constexpr int operator[](std::size_t i) const noexcept { return coords[i]; }
    constexpr int *begin() noexcept { return coords; }
    constexpr int *end() noexcept { return coords + D; }
    [[nodiscard]] constexpr const int *begin() const noexcept { return coords; }
    [[nodiscard]] constexpr const int *end() const noexcept { return coords + D; }

    friend constexpr bool operator==(const lattice_point &, const lattice_point &) = default;
    friend constexpr auto operator<=>(const lattice_point &, const lattice_point &) = default;

    friend constexpr lattice_point operator+(lattice_point a, const lattice_point &b) noexcept {
        for (int i = 0; i < D; ++i) {
            a.coords[i] += b.coords[i];
        }
        return a;
    }
    friend constexpr lattice_point operator-(lattice_point a, const lattice_point &b) noexcept {
        for (int i = 0; i < D; ++i) {
            a.coords[i] -= b.coords[i];
        }
        return a;
    }
};

} // namespace num

template <int D>
struct std::hash<num::lattice_point<D>> {
    std::size_t operator()(const num::lattice_point<D> &x) const noexcept {
        std::size_t seed = D;
        for (int i = 0; i < D; ++i) {
            seed ^= static_cast<std::size_t>(x.coords[i]) + 0x9e3779b9U + (seed << 6) + (seed >> 2);
        }
        return seed;
    }
};
