/// @file container/concepts.hpp
/// @brief Storage-layout concepts: contiguous, row-major, CSR, banded, tridiagonal.
#pragma once

#include "algebra/concepts.hpp"
#include "algebra/scalar.hpp"
#include "core/types.hpp"
#include <concepts>
#include <type_traits>

namespace num {

// 2. Representation

/// @brief Storage-layout concepts, for kernels that need pointers and strides.
namespace repr {

/// @brief Contiguous vector: elements reachable through `data()`.
template <class V>
concept contiguous = vector_space<V> && requires(V v, const V cv) {
    { cv.data() } -> std::convertible_to<const scalar_t<V> *>;
    { v.data() } -> std::convertible_to<scalar_t<V> *>;
};

/// @brief Row-major dense matrix: contiguous, row stride `cols()`.
template <class A>
concept dense_row_major = matrix_space<A> && requires(A a, const A ca) {
    { ca.data() } -> std::convertible_to<const entry_t<A> *>;
    { a.data() } -> std::convertible_to<entry_t<A> *>;
};

/// @brief CSR sparse matrix: row offsets, column indices and values.
template <class M>
concept csr = requires(const M &m) {
    { m.n_rows() } -> std::convertible_to<idx>;
    { m.n_cols() } -> std::convertible_to<idx>;
    { m.nnz() } -> std::convertible_to<idx>;
    { m.row_ptr() } -> std::convertible_to<const idx *>;
    { m.col_idx() } -> std::convertible_to<const idx *>;
    { m.values() };
};

/// @brief Banded matrix: entries confined to \f$-k_l \le j - i \le k_u\f$.
template <class B>
concept banded = matrix_space<B> && (requires(const B &b) {
    { b.kl() } -> std::convertible_to<idx>;
    { b.ku() } -> std::convertible_to<idx>;
} || requires(const B &b) {
    { b.lower_bandwidth() } -> std::convertible_to<idx>;
    { b.upper_bandwidth() } -> std::convertible_to<idx>;
});

/// @brief Lower bandwidth \f$k_l\f$ of a banded matrix.
template <class B>
[[nodiscard]] constexpr idx lower_bandwidth_of(const B &b) {
    if constexpr (requires { b.kl(); }) {
        return b.kl();
    } else {
        return b.lower_bandwidth();
    }
}

/// @brief Upper bandwidth \f$k_u\f$ of a banded matrix.
template <class B>
[[nodiscard]] constexpr idx upper_bandwidth_of(const B &b) {
    if constexpr (requires { b.ku(); }) {
        return b.ku();
    } else {
        return b.upper_bandwidth();
    }
}

/// @brief Tridiagonal matrix: sub-, main and superdiagonals `dl`, `d`, `du`.
template <class T>
concept tridiagonal = requires(const T &t) {
    { t.dl };
    { t.d };
    { t.du };
    { t.size() } -> std::convertible_to<idx>;
};

} // namespace repr

} // namespace num
