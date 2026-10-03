/// @file linear/factorization/factor.hpp
/// @brief Factor a matrix once and solve against any number of right-hand sides.
#pragma once

#include "linear/factorization/block_tridiagonal.hpp"
#include "linear/factorization/cholesky.hpp"
#include "linear/factorization/inverse_diagonal.hpp"
#include "linear/factorization/lu.hpp"
#include "linear/solve.hpp"
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

/// @brief A factorization of the diagonal similarity of a matrix under the weights `h`, with
/// the weights it used.
template <class F>
struct similar_factor {
    F factor;
    vec<real> h;
};

namespace detail {

[[nodiscard]] inline vec<real> checked_weights(view<const real> h) {
    vec<real> result(h.size(), 0.0);
    for (idx i = 0; i < h.size(); ++i) {
        if (!(h[i] > 0.0)) {
            throw std::invalid_argument("factor: similarity weights must be positive");
        }
        result[i] = h[i];
    }
    return result;
}

[[nodiscard]] inline vec<real> reciprocal(view<const real> h) {
    vec<real> result(h.size(), 0.0);
    for (idx i = 0; i < h.size(); ++i) {
        result[i] = 1.0 / h[i];
    }
    return result;
}

template <class RHS>
inline void scale_rows(RHS &x, view<const real> h, bool inverse) {
    if constexpr (std::is_same_v<RHS, vec<real>>) {
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

} // namespace detail

/// Factor a sparse unstructured nonsingular M-matrix by no-pivot LU.
/// @throws std::runtime_error If a pivot is zero.
[[nodiscard]] inline lu_result<real> lu(const spmat &R, no_pivot_structure) {
    lu_result<real> Z = lu(dense(R), no_pivot);
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
    vec<real> weights = detail::checked_weights(h);
    cholesky_result C =
        cholesky(assume_spd(dense(sparse_diagonal_similarity(R, detail::reciprocal(weights)))));
    if (!C.positive_definite) {
        throw std::runtime_error("cholesky: symmetrized matrix is not positive definite");
    }
    return {std::move(C), std::move(weights)};
}

/// Block-Cholesky specialization of the diagonally similar factorization.
[[nodiscard]] inline similar_factor<block_cholesky_factor>
cholesky(const spmat &R, block_structure structure, view<const real> h) {
    vec<real> weights = detail::checked_weights(h);
    block_cholesky_factor C = factor_block_cholesky(
        sparse_diagonal_similarity(R, detail::reciprocal(weights)), structure.levels);
    return {std::move(C), std::move(weights)};
}

inline void solve_transpose(const block_cholesky_factor &Z, const vec<real> &b, vec<real> &x) {
    solve(Z, b, x);
}
inline void solve_transpose(const block_cholesky_factor &Z, const mat<real> &B, mat<real> &X) {
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

template <class F>
[[nodiscard]] vec<real> inverse_diagonal(const F &Z, idx n) {
    vec<real> diagonal(n, 0.0);
    constexpr idx columns_per_solve = 64;
    for (idx first = 0; first < n; first += columns_per_solve) {
        const idx count = std::min(columns_per_solve, n - first);
        mat<real> E(n, count, 0.0);
        for (idx column = 0; column < count; ++column) {
            E(first + column, column) = 1.0;
        }
        const mat<real> ZE = solve(Z, E);
        for (idx column = 0; column < count; ++column) {
            diagonal[first + column] = ZE(first + column, column);
        }
    }
    return diagonal;
}

} // namespace num
