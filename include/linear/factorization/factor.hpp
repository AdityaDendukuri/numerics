/// @file linear/factorization/factor.hpp
/// @brief Factor a matrix once and solve against any number of right-hand sides.
#pragma once

#include "linear/factorization/block_tridiagonal.hpp"
#include "linear/factorization/cholesky.hpp"
#include "linear/factorization/inverse_diagonal.hpp"
#include "linear/factorization/lu_no_pivot.hpp"
#include "linear/matrix_utils.hpp"
#include "linear/solvers/auto_linear.hpp"
#include "linear/sparse/sparse.hpp"
#include <algorithm>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace num {

/// @brief The block-level structure of a matrix, one level label per row, for `num::lu` and
/// `num::cholesky` on block-tridiagonal matrices.
struct block_structure {
    view<const idx> levels;
};

/// @brief Select block-tridiagonal factorization with one level label per row.
[[nodiscard]] inline block_structure blocks(view<const idx> levels) {
    return {levels};
}

/// @brief The tag type of `num::sparse`.
struct sparse_structure {};
/// @brief Select sparse direct factorization.
inline constexpr sparse_structure sparse{};

/// @brief The tag type of `num::no_pivot`.
struct no_pivot_structure {};
/// @brief Select dense LU without pivoting, for matrices whose structure keeps the pivots
/// nonzero.
inline constexpr no_pivot_structure no_pivot{};

/// @brief A factorization of the diagonal similarity of a matrix under the weights `h`, with
/// the weights it used.
template <class F>
struct similar_factor {
    F factor;
    vec h;
};

/// @brief How many blocks a block factorization kept from the previous one, and how many rows
/// they cover.
struct suffix_reuse_report {
    idx blocks = 0;
    idx reused_blocks = 0;
    idx reused_rows = 0;
};

/// @brief Returned when no leading block of a factorization can be kept.
inline constexpr idx no_reusable_block = static_cast<idx>(-1);

namespace detail {

[[nodiscard]] inline vec checked_weights(view<const real> h) {
    vec result(h.size(), 0.0);
    for (idx i = 0; i < h.size(); ++i) {
        if (!(h[i] > 0.0)) {
            throw std::invalid_argument("factor: similarity weights must be positive");
        }
        result[i] = h[i];
    }
    return result;
}

[[nodiscard]] inline vec reciprocal(view<const real> h) {
    vec result(h.size(), 0.0);
    for (idx i = 0; i < h.size(); ++i) {
        result[i] = 1.0 / h[i];
    }
    return result;
}

template <class RHS>
inline void scale_rows(RHS &x, view<const real> h, bool inverse) {
    if constexpr (std::is_same_v<RHS, vec>) {
        for (idx i = 0; i < x.size(); ++i) {
            x[i] *= inverse ? 1.0 / h[i] : h[i];
        }
    } else {
        for (idx i = 0; i < x.rows(); ++i) {
            for (idx j = 0; j < x.cols(); ++j) {
                x(i, j) *= inverse ? 1.0 / h[i] : h[i];
            }
        }
    }
}

template <typename F>
[[nodiscard]] inline idx first_changed_block(const F &Z, const block_layout &layout,
                                             view<const idx> changed) {
    array<idx> old_position(Z.size);
    for (idx position = 0; position < Z.size; ++position) {
        old_position[Z.order[position]] = position;
    }

    idx first = layout.offsets.size() - 1;
    for (idx row : changed) {
        first = std::min(first, layout.block_of[row]);
        const auto boundary =
            std::upper_bound(Z.offsets.begin(), Z.offsets.end(), old_position[row]);
        first = std::min(first, static_cast<idx>(boundary - Z.offsets.begin() - 1));
    }
    if (first > Z.blocks() || first + 1 > layout.offsets.size()) {
        return no_reusable_block;
    }
    for (idx k = 0; k <= first; ++k) {
        if (Z.offsets[k] != layout.offsets[k]) {
            return no_reusable_block;
        }
    }
    for (idx position = 0; position < layout.offsets[first]; ++position) {
        if (Z.order[position] != layout.order[position]) {
            return no_reusable_block;
        }
    }
    return first;
}

inline void record_suffix_reuse(suffix_reuse_report *report, const block_layout &layout,
                                idx first) {
    if (report != nullptr) {
        *report = {.blocks = static_cast<idx>(layout.offsets.size() - 1),
                   .reused_blocks = first,
                   .reused_rows = layout.offsets[first]};
    }
}

} // namespace detail

/// Factor a dense unstructured nonsingular M-matrix by no-pivot LU.
[[nodiscard]] inline no_pivot_lu lu(const mat &R, no_pivot_structure) {
    no_pivot_lu Z = factor_no_pivot(R);
    if (Z.singular) {
        throw std::runtime_error("lu: matrix is singular");
    }
    return Z;
}

/// Factor a sparse unstructured nonsingular M-matrix by no-pivot LU.
[[nodiscard]] inline no_pivot_lu lu(const spmat &R, no_pivot_structure) {
    no_pivot_lu Z = factor_no_pivot(dense(R));
    if (Z.singular) {
        throw std::runtime_error("lu: matrix is singular");
    }
    return Z;
}

/// Factor a block-tridiagonal nonsingular M-matrix by block LU.
[[nodiscard]] inline block_lu_factor lu(const spmat &R, block_structure structure) {
    return factor_block_lu(R, structure.levels);
}

/// Factor a large unstructured matrix with the configured sparse backend.
[[nodiscard]] inline auto_linear_solver lu(const spmat &R, sparse_structure) {
    return auto_linear_solver(R);
}

/// Factor H R H^-1 by Cholesky and retain H for applications of R^-1.
[[nodiscard]] inline similar_factor<cholesky_result> cholesky(const spmat &R, view<const real> h) {
    vec weights = detail::checked_weights(h);
    cholesky_result C =
        cholesky(assume_spd(dense(sparse_diagonal_similarity(R, detail::reciprocal(weights)))));
    if (!C.success) {
        throw std::runtime_error("cholesky: symmetrized matrix is not positive definite");
    }
    return {std::move(C), std::move(weights)};
}

/// Block-Cholesky specialization of the diagonally similar factorization.
[[nodiscard]] inline similar_factor<block_cholesky_factor>
cholesky(const spmat &R, block_structure structure, view<const real> h) {
    vec weights = detail::checked_weights(h);
    block_cholesky_factor C = factor_block_cholesky(
        sparse_diagonal_similarity(R, detail::reciprocal(weights)), structure.levels);
    return {std::move(C), std::move(weights)};
}

/// Compatibility wrappers for code written before the algorithm names were explicit.
[[nodiscard]] inline no_pivot_lu factor(const spmat &R) {
    return lu(R, no_pivot);
}
[[nodiscard]] inline block_lu_factor factor(const spmat &R, block_structure structure) {
    return lu(R, structure);
}
[[nodiscard]] inline auto_linear_solver factor(const spmat &R, sparse_structure structure) {
    return lu(R, structure);
}
[[nodiscard]] inline similar_factor<cholesky_result> factor(const spmat &R, view<const real> h) {
    return cholesky(R, h);
}
[[nodiscard]] inline similar_factor<block_cholesky_factor>
factor(const spmat &R, block_structure structure, view<const real> h) {
    return cholesky(R, structure, h);
}

/// Refactor the affected suffix of a block LU factorization.
[[nodiscard]] inline std::optional<block_lu_factor>
refactor_suffix(const block_lu_factor &Z, const spmat &R, block_structure structure,
                view<const idx> changed, suffix_reuse_report *report = nullptr) {
    if (R.n_rows() != Z.size || structure.levels.size() != Z.size) {
        return std::nullopt;
    }
    const detail::block_layout layout = detail::build_block_order(structure.levels);
    const idx first = detail::first_changed_block(Z, layout, changed);
    if (first == no_reusable_block) {
        return std::nullopt;
    }
    detail::record_suffix_reuse(report, layout, first);
    try {
        return refactor_block_lu_suffix(R, structure.levels, Z, first);
    } catch (const std::exception &) {
        return std::nullopt;
    }
}

/// Refactor the affected suffix of a diagonally similar block Cholesky factorization.
[[nodiscard]] inline std::optional<similar_factor<block_cholesky_factor>>
refactor_suffix(const similar_factor<block_cholesky_factor> &Z, const spmat &R,
                block_structure structure, view<const real> h, view<const idx> changed,
                suffix_reuse_report *report = nullptr) {
    if (R.n_rows() != Z.factor.size || structure.levels.size() != Z.factor.size) {
        return std::nullopt;
    }
    const detail::block_layout layout = detail::build_block_order(structure.levels);
    const idx first = detail::first_changed_block(Z.factor, layout, changed);
    if (first == no_reusable_block) {
        return std::nullopt;
    }
    detail::record_suffix_reuse(report, layout, first);
    try {
        vec weights = detail::checked_weights(h);
        auto C = refactor_block_cholesky_suffix(
            sparse_diagonal_similarity(R, detail::reciprocal(weights)), structure.levels, Z.factor,
            first);
        return similar_factor<block_cholesky_factor>{std::move(C), std::move(weights)};
    } catch (const std::exception &) {
        return std::nullopt;
    }
}

inline void solve(const cholesky_result &Z, const vec &b, vec &x) {
    cholesky_solve(Z, b, x);
}
inline void solve(const cholesky_result &Z, const mat &B, mat &X) {
    cholesky_solve(Z, B, X);
}
inline void solve_transpose(const cholesky_result &Z, const vec &b, vec &x) {
    cholesky_solve(Z, b, x);
}
inline void solve_transpose(const cholesky_result &Z, const mat &B, mat &X) {
    cholesky_solve(Z, B, X);
}

inline void solve_transpose(const block_cholesky_factor &Z, const vec &b, vec &x) {
    solve(Z, b, x);
}
inline void solve_transpose(const block_cholesky_factor &Z, const mat &B, mat &X) {
    solve(Z, B, X);
}

template <class F, class RHS>
inline void solve(const similar_factor<F> &Z, const RHS &b, RHS &x) {
    x = b;
    detail::scale_rows(x, Z.h, false);
    solve(Z.factor, x, x);
    detail::scale_rows(x, Z.h, true);
}

template <class F, class RHS>
inline void solve_transpose(const similar_factor<F> &Z, const RHS &b, RHS &x) {
    x = b;
    detail::scale_rows(x, Z.h, true);
    solve(Z.factor, x, x);
    detail::scale_rows(x, Z.h, false);
}

namespace detail {

template <class F>
struct transposed_factor {
    const F &base;
};

} // namespace detail

/// Write a transposed solve with the same `solve` operation used otherwise.
/// The view stores only a reference, so this does not transpose or copy the factors.
template <class F>
requires requires(const F &Z, const vec &b, vec &x) {
    solve_transpose(Z, b, x);
}
[[nodiscard]] inline detail::transposed_factor<F> transpose(const F &Z) {
    return {Z};
}

template <class F>
requires(!std::is_lvalue_reference_v<F> &&
         requires(const std::remove_reference_t<F> &Z, const vec &b, vec &x) {
             solve_transpose(Z, b, x);
         }) detail::transposed_factor<std::remove_reference_t<F>> transpose(F &&) = delete;

template <class F>
[[nodiscard]] inline const F &transpose(detail::transposed_factor<F> Z) {
    return Z.base;
}

template <class F, class RHS>
inline void solve(detail::transposed_factor<F> Z, const RHS &b, RHS &x) {
    solve_transpose(Z.base, b, x);
}

template <class F, class RHS>
[[nodiscard]] RHS solve(const F &Z, const RHS &b) requires requires(RHS &x) {
    solve(Z, b, x);
}
{
    RHS x;
    solve(Z, b, x);
    return x;
}

template <class F>
[[nodiscard]] vec inverse_diagonal(const F &Z, idx n) {
    vec diagonal(n, 0.0);
    constexpr idx columns_per_solve = 64;
    for (idx first = 0; first < n; first += columns_per_solve) {
        const idx count = std::min(columns_per_solve, n - first);
        mat E(n, count, 0.0);
        for (idx column = 0; column < count; ++column) {
            E(first + column, column) = 1.0;
        }
        const mat ZE = solve(Z, E);
        for (idx column = 0; column < count; ++column) {
            diagonal[first + column] = ZE(first + column, column);
        }
    }
    return diagonal;
}

} // namespace num
