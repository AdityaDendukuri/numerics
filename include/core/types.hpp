/// @file types.hpp
/// @brief Core type definitions
#pragma once

#include <array>
#include <complex>
#include <cstddef>
#include <functional>
#include <map>
#include <span>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace num {

using real = double;
using idx = std::size_t;
using cplx = std::complex<real>;

/// @name Container vocabulary
///
/// Four standard containers are named after things that already mean something
/// else in linear algebra, so code mixing them with this library reads
/// ambiguously:
///
/// | standard name        | what the word means here instead              |
/// |----------------------|-----------------------------------------------|
/// | `std::vector`        | @ref num::vec, an element of a vector space   |
/// | `std::span`          | the span of a set of vectors, i.e. a subspace |
/// | `std::map`           | a linear map                                  |
/// | `std::unordered_map` | likewise                                      |
///
/// The aliases below name those containers after what they are rather than
/// after a word the library has already spent, so a declaration says at a
/// glance whether it is about storage or about mathematics:
///
/// ```cpp
/// num::array<num::idx> row_offsets;   // storage
/// num::vec             x(n);          // mathematics
/// ```
///
/// These are alias templates, not wrappers. `num::array<T>` *is*
/// `std::vector<T>` -- the same type, no conversion anywhere, accepted
/// unchanged by every standard algorithm and every third-party signature that
/// takes a `std::vector`. Nothing is hidden and nothing is re-implemented; only
/// the spelling differs. Compiler diagnostics still name the standard type.
///
/// Fixed and dynamic extent are two names rather than one name with an extent
/// parameter, because an alias that selected between them through a trait would
/// resolve to a dependent qualified name -- a non-deduced context -- and
/// `template <class T> void f(array<T> &)` would then fail to deduce `T`.
///
/// Containers whose names carry no mathematical meaning -- `std::pair`,
/// `std::tuple`, `std::optional`, `std::string` -- are deliberately left alone.
/// Renaming those buys nothing and would oblige the library to maintain a
/// parallel vocabulary for the whole standard library.
///
/// For numeric data prefer @ref num::vec over `num::array<real>`: it owns
/// over-aligned storage, skips the zero-initialising pass on construction, and
/// satisfies @ref num::math::vector_space, so the solvers take it directly.
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

/// @brief A hash table from keys to values.
template <class K, class V, class Hash = std::hash<K>, class Eq = std::equal_to<K>,
          class Alloc = std::allocator<std::pair<const K, V>>>
using table = std::unordered_map<K, V, Hash, Eq, Alloc>;

/// @brief A table kept in key order, for iteration from smallest key to largest.
template <class K, class V, class Compare = std::less<K>,
          class Alloc = std::allocator<std::pair<const K, V>>>
using sorted_table = std::map<K, V, Compare, Alloc>;

/// @brief A hash table of unique keys.
template <class K, class Hash = std::hash<K>, class Eq = std::equal_to<K>,
          class Alloc = std::allocator<K>>
using key_set = std::unordered_set<K, Hash, Eq, Alloc>;

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
