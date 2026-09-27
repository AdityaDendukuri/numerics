/// @file types.hpp
/// @brief Core type definitions
#pragma once

#include <array>
#include <complex>
#include <cstddef>
#include <functional>
#include <map>
#include <set>
#include <span>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace num {

using real = double;
using idx = std::size_t;
using cplx = std::complex<real>;

/// @name Container vocabulary
///
/// Alias templates, not wrappers: `num::array<T>` is `std::vector<T>`. `array`, `static_array`
/// and `view` free the words vector and span for mathematics; the associative containers keep
/// their C++ names. See the containers page, section 1.
///
/// Fixed and dynamic extent get two names because an alias selecting between them through a
/// trait would make `template <class T> void f(array<T> &)` a non-deduced context.
/// @{

/// @brief A growable array. Storage, not an element of a vector space.
template <class T, class Alloc = std::allocator<T>>
using array = std::vector<T, Alloc>;

/// @brief An array whose length is fixed at compile time.
template <class T, std::size_t N>
using static_array = std::array<T, N>;

/// @brief A non-owning window onto contiguous storage.
template <class T, std::size_t Extent = std::dynamic_extent>
using view = std::span<T, Extent>;

/// @brief `std::unordered_map`: a hash map with no key order, like Rust `HashMap`.
template <class K, class V, class Hash = std::hash<K>, class Eq = std::equal_to<K>,
          class Alloc = std::allocator<std::pair<const K, V>>>
using unordered_map = std::unordered_map<K, V, Hash, Eq, Alloc>;

/// @brief `std::map`: a map sorted by key, like Rust `BTreeMap`.
template <class K, class V, class Compare = std::less<K>,
          class Alloc = std::allocator<std::pair<const K, V>>>
using map = std::map<K, V, Compare, Alloc>;

/// @brief `std::unordered_set`: a hash set with no key order, like Rust `HashSet`.
template <class K, class Hash = std::hash<K>, class Eq = std::equal_to<K>,
          class Alloc = std::allocator<K>>
using unordered_set = std::unordered_set<K, Hash, Eq, Alloc>;

/// @brief `std::set`: a set sorted by key, like Rust `BTreeSet`.
template <class K, class Compare = std::less<K>, class Alloc = std::allocator<K>>
using set = std::set<K, Compare, Alloc>;

/// @brief Add an element at the end of an array, constructed from `args`. It is `emplace_back`,
/// named for what the caller does, and returns a reference to the new element.
template <class T, class Alloc, class... Args>
T &append(array<T, Alloc> &values, Args &&...args) {
    return values.emplace_back(std::forward<Args>(args)...);
}

/// @}

/// @brief Cast any integer to idx without a verbose static_cast.
template <class T>
constexpr idx to_idx(T x) noexcept {
    return static_cast<idx>(x);
}

/// scalar callback \f$f(x)\f$.
using scalar_fn = std::function<real(real)>;

/// vec callback writing \f$f(t, y)\f$ into a caller-provided buffer.
using vector_fn = std::function<void(real, real *, real *)>;

} // namespace num
