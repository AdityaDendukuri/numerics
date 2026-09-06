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

/// @brief Storage-layout predicates describing contiguous memory layouts and sparse formats.
///
/// These representation concepts are structural constraints used by low-level hardware kernels,
/// BLAS/LAPACK wrappers, and memory allocators.
namespace repr {

/// @brief vec stored in one contiguous block, addressable via a direct raw pointer `.data()`.
///
/// Models include `num::vec`, `num::cvec`, and `std::vector<double>`.
///
/// @tparam V Container type.
template <class V>
concept contiguous = vector_space<V> && requires(V v, const V cv) {
    { cv.data() } -> std::convertible_to<const scalar_t<V> *>;
    { v.data() } -> std::convertible_to<scalar_t<V> *>;
};

/// @brief mat stored contiguously in row-major order with row stride equal to `cols()`.
///
/// Allows zero-copy lowering directly to CBLAS row-major routines (`cblas_dgemm`, `cblas_dgemv`).
/// Models include `num::mat`.
///
/// @tparam A mat type.
template <class A>
concept dense_row_major = matrix_space<A> && requires(A a, const A ca) {
    { ca.data() } -> std::convertible_to<const entry_t<A> *>;
    { a.data() } -> std::convertible_to<entry_t<A> *>;
};

/// @brief Compressed sparse row (CSR) storage format: row offsets, column indices, and nonzero values.
///
/// Enables matrix-vector multiplication in \f$\mathcal{O}(\text{nnz})\f$ time.
/// Models include `num::spmat`.
///
/// @tparam M Sparse matrix type.
template <class M>
concept csr = requires(const M &m) {
    { m.n_rows() } -> std::convertible_to<idx>;
    { m.n_cols() } -> std::convertible_to<idx>;
    { m.nnz() } -> std::convertible_to<idx>;
    { m.row_ptr() } -> std::convertible_to<const idx *>;
    { m.col_idx() } -> std::convertible_to<const idx *>;
    { m.values() };
};

/// @brief banded storage format: entries confined to \f$-k_l \leq j - i \leq k_u\f$.
///
/// Models include `num::band_mat`.
///
/// @tparam B banded matrix type.
template <class B>
concept banded = matrix_space<B> && (requires(const B &b) {
    { b.kl() } -> std::convertible_to<idx>;
    { b.ku() } -> std::convertible_to<idx>;
} || requires(const B &b) {
    { b.lower_bandwidth() } -> std::convertible_to<idx>;
    { b.upper_bandwidth() } -> std::convertible_to<idx>;
});

/// @brief Lower bandwidth \f$k_l\f$ of a banded matrix under either accessor spelling.
template <class B>
[[nodiscard]] constexpr idx lower_bandwidth_of(const B &b) {
    if constexpr (requires { b.kl(); }) {
        return b.kl();
    } else {
        return b.lower_bandwidth();
    }
}

/// @brief Upper bandwidth \f$k_u\f$ of a banded matrix under either accessor spelling.
template <class B>
[[nodiscard]] constexpr idx upper_bandwidth_of(const B &b) {
    if constexpr (requires { b.ku(); }) {
        return b.ku();
    } else {
        return b.upper_bandwidth();
    }
}

/// @brief tridiagonal storage format: subdiagonal (`dl`), main diagonal (`d`), and superdiagonal (`du`).
///
/// @tparam T tridiagonal matrix type.
template <class T>
concept tridiagonal = requires(const T &t) {
    { t.dl };
    { t.d };
    { t.du };
    { t.size() } -> std::convertible_to<idx>;
};

} // namespace repr

} // namespace num
