/// @file container/concepts.hpp
/// @brief Representation: how a container stores its elements.
///
/// Storage-layout predicates, kept deliberately apart from the algebraic structure
/// in `core/concepts.hpp`. CSR is a memory format, not a property of a linear map.
/// These are the right constraint for a kernel that needs a pointer and a stride,
/// and the wrong one for an algorithm that needs a vector space.
#pragma once

#include "algebra/concepts.hpp"
#include "algebra/scalar.hpp"
#include "core/types.hpp"
#include <concepts>
#include <type_traits>

namespace num {

// =============================================================================
// 2. Representation
// =============================================================================

/// @brief Storage-layout predicates. These describe memory, not mathematics.
///
/// They are the right constraint for a kernel that needs a pointer and a stride,
/// and the wrong constraint for an algorithm that needs a vector space.
namespace repr {

/// @brief Vector stored in one contiguous block, addressable as a raw pointer.
template <class V>
concept Contiguous = VectorSpace<V> && requires(V v, const V cv) {
    { cv.data() } -> std::convertible_to<const scalar_t<V> *>;
    { v.data() } -> std::convertible_to<scalar_t<V> *>;
};

/// @brief Matrix stored contiguously in row-major order.
template <class A>
concept DenseRowMajor = MatrixSpace<A> && requires(A a, const A ca) {
    { ca.data() } -> std::convertible_to<const entry_t<A> *>;
    { a.data() } -> std::convertible_to<entry_t<A> *>;
};

/// @brief Compressed sparse row storage: row offsets, column indices, values.
template <class M>
concept CSR = requires(const M &m) {
    { m.n_rows() } -> std::convertible_to<idx>;
    { m.n_cols() } -> std::convertible_to<idx>;
    { m.nnz() } -> std::convertible_to<idx>;
    { m.row_ptr() } -> std::convertible_to<const idx *>;
    { m.col_idx() } -> std::convertible_to<const idx *>;
    { m.values() };
};

/// @brief Banded storage: entries confined to \f$-k_l \leq j - i \leq k_u\f$.
///
/// Accepts either spelling of the bandwidth accessors. `num::BandedMatrix` names
/// them `kl()`/`ku()` after the LAPACK convention; the long form is offered for
/// foreign types. Requiring only the long form, as this concept once did, made it
/// unsatisfiable by the library's own banded matrix.
template <class B>
concept Banded = MatrixSpace<B> && (requires(const B &b) {
    { b.kl() } -> std::convertible_to<idx>;
    { b.ku() } -> std::convertible_to<idx>;
} || requires(const B &b) {
    { b.lower_bandwidth() } -> std::convertible_to<idx>;
    { b.upper_bandwidth() } -> std::convertible_to<idx>;
});

/// @brief Lower bandwidth of a banded matrix, under either accessor spelling.
template <class B>
[[nodiscard]] constexpr idx lower_bandwidth_of(const B &b) {
    if constexpr (requires { b.kl(); }) {
        return b.kl();
    } else {
        return b.lower_bandwidth();
    }
}

/// @brief Upper bandwidth of a banded matrix, under either accessor spelling.
template <class B>
[[nodiscard]] constexpr idx upper_bandwidth_of(const B &b) {
    if constexpr (requires { b.ku(); }) {
        return b.ku();
    } else {
        return b.upper_bandwidth();
    }
}

/// @brief Tridiagonal storage: the three occupied diagonals held separately.
template <class T>
concept Tridiagonal = requires(const T &t) {
    { t.dl };
    { t.d };
    { t.du };
    { t.size() } -> std::convertible_to<idx>;
};

} // namespace repr

} // namespace num
