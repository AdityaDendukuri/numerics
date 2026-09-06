/// @file linear/factorization/block_tridiagonal.hpp
/// @brief Block LU and block Cholesky for block-tridiagonal sparse matrices.
///
/// A matrix whose rows carry a *level* label is block tridiagonal when every
/// nonzero couples rows of the same level or of adjacent levels. Reordering by
/// level makes that structure explicit, and the factorization then proceeds one
/// block row at a time: each step touches only a dense diagonal block and its two
/// neighbours, so the work is \f$O(\sum_k n_k^3)\f$ rather than \f$O(n^3)\f$, and
/// no fill appears outside the band.
///
/// The level labels are the caller's. This header takes them as given, compresses
/// them to contiguous block indices, and never interprets them: what a level
/// *means* — a copy number, a mesh layer, a time slice — belongs to the caller,
/// as does the choice between LU and Cholesky.
///
/// ### Index convention
///
/// With `nb` blocks, `offsets` has `nb + 1` entries and block `k` spans the
/// reordered rows `[offsets[k], offsets[k+1])`. Both `upper` and `lower` have
/// `nb - 1` entries and are indexed by the *coupling*, not the block row:
///
///   - `upper[k]` is the block at row `k`, column `k+1`;
///   - `lower[k]` is the block at row `k+1`, column `k`.
///
/// So the elimination reads `lower[k] <- lower[k] D_k^{-1}` followed by
/// `D_{k+1} <- D_{k+1} - lower[k] upper[k]`.
///
/// ### The factors are public
///
/// `block_lu_factor` and `block_cholesky_factor` are plain aggregates rather than the
/// internals of a solver object, so a later low-rank layer can reach the block
/// factors directly: apply the factor to a tall `U`, apply its transpose to `V`,
/// form the small Woodbury matrix \f$K = I + V^{T}A_0^{-1}U\f$, and keep reusing
/// the same block factor until the update rank or residual makes that
/// unprofitable. Nothing here needs to change for that to work.
///
/// All dense arithmetic lowers to the existing kernels — `num::factor_no_pivot`,
/// `num::cholesky`, `num::matmul`, `num::transpose`, and the raw triangular
/// solves. No GEMM or triangular solve is reimplemented.
#pragma once

#include "container/matrix.hpp"
#include "container/matrix_ops.hpp"
#include "container/vector.hpp"
#include "core/types.hpp"
#include "kernel/kernel.hpp"
#include "linear/factorization/cholesky.hpp"
#include "linear/factorization/lu.hpp"
#include "linear/factorization/lu_no_pivot.hpp"
#include "linear/matrix_properties.hpp"
#include "linear/matrix_utils.hpp"
#include "linear/sparse/sparse.hpp"
#include <algorithm>
#include <span>
#include <stdexcept>
#include <vector>

namespace num {

/// @brief Block LU factors of a block-tridiagonal matrix.
///
/// After factorization the matrix reads \f$PA P^{T} = \mathcal{L}\,\mathcal{U}\f$
/// with \f$\mathcal{L}\f$ block unit lower bidiagonal (its off-diagonal blocks in
/// `lower`) and \f$\mathcal{U}\f$ block upper bidiagonal (`diagonal` on the
/// diagonal, `upper` off it).
struct block_lu_factor {
    idx size = 0;                  ///< Order of the original matrix.
    array<idx> offsets;      ///< `nb + 1` block boundaries in reordered indexing.
    array<idx> order;        ///< `order[p]` is the original row at reordered position `p`.
    array<no_pivot_lu> diagonal; ///< No-pivot factors of the diagonal Schur blocks.
    array<mat> upper;     ///< `upper[k]`: block (k, k+1). Size `nb - 1`.
    array<mat> lower;     ///< `lower[k]`: block (k+1, k), already scaled. Size `nb - 1`.

    /// @brief Number of blocks.
    [[nodiscard]] idx blocks() const noexcept {
        return offsets.empty() ? 0 : offsets.size() - 1;
    }
    /// @brief Rows in block `k`.
    [[nodiscard]] idx block_size(idx k) const { return offsets[k + 1] - offsets[k]; }
    /// @brief True when some diagonal block was singular.
    [[nodiscard]] bool singular() const noexcept {
        return std::any_of(diagonal.begin(), diagonal.end(),
                           [](const no_pivot_lu &f) { return f.singular; });
    }
};

/// @brief Block Cholesky factors of a symmetric positive-definite block-tridiagonal matrix.
///
/// \f$PAP^{T} = \mathcal{L}\mathcal{L}^{T}\f$ with \f$\mathcal{L}\f$ block lower
/// bidiagonal: `diagonal[k]` holds the dense Cholesky factor of block `k`, and
/// `lower[k]` the block at row `k+1`, column `k`, already right-scaled by
/// \f$L_k^{-T}\f$.
struct block_cholesky_factor {
    idx size = 0;
    array<idx> offsets;
    array<idx> order;
    array<cholesky_result> diagonal;
    array<mat> lower; ///< `lower[k]`: block (k+1, k). Size `nb - 1`.

    [[nodiscard]] idx blocks() const noexcept {
        return offsets.empty() ? 0 : offsets.size() - 1;
    }
    [[nodiscard]] idx block_size(idx k) const { return offsets[k + 1] - offsets[k]; }
    /// @brief True when some diagonal block was not positive definite.
    [[nodiscard]] bool failed() const noexcept {
        return std::any_of(diagonal.begin(), diagonal.end(),
                           [](const cholesky_result &f) { return !f.success; });
    }
};

namespace detail {

/// @brief Reordering induced by a level labelling.
struct block_layout {
    array<idx> offsets;  ///< `nb + 1` boundaries.
    array<idx> order;    ///< reordered position -> original row.
    array<idx> position; ///< original row -> reordered position.
    array<idx> block_of; ///< original row -> block index.
};

/// @brief Compress arbitrary level labels into contiguous blocks and build the permutation.
///
/// Labels need not be contiguous, zero-based, or dense; they are sorted, and the
/// distinct values become blocks `0, 1, ...` in that order. Rows keep their
/// original relative order within a block, so the permutation is deterministic
/// and a run of already-sorted labels is the identity.
[[nodiscard]] inline block_layout build_block_order(view<const idx> levels) {
    const idx n = levels.size();
    array<idx> distinct(levels.begin(), levels.end());
    std::sort(distinct.begin(), distinct.end());
    distinct.erase(std::unique(distinct.begin(), distinct.end()), distinct.end());

    block_layout layout;
    const idx blocks = distinct.size();
    layout.block_of.resize(n);
    layout.position.resize(n);
    layout.order.resize(n);
    layout.offsets.assign(blocks + 1, 0);

    for (idx row = 0; row < n; ++row) {
        const auto found = std::lower_bound(distinct.begin(), distinct.end(), levels[row]);
        layout.block_of[row] = static_cast<idx>(found - distinct.begin());
        layout.offsets[layout.block_of[row] + 1] += 1;
    }
    for (idx k = 0; k < blocks; ++k) {
        layout.offsets[k + 1] += layout.offsets[k];
    }

    array<idx> cursor(layout.offsets.begin(), layout.offsets.end() - 1);
    for (idx row = 0; row < n; ++row) {
        const idx slot = cursor[layout.block_of[row]]++;
        layout.order[slot] = row;
        layout.position[row] = slot;
    }
    return layout;
}

/// @brief Reject any nonzero coupling non-adjacent blocks.
///
/// Structural entries that store an exact zero are skipped: they contribute
/// nothing to the factorization, and rejecting them would make an explicitly
/// padded sparsity pattern unusable for no numerical reason.
inline void validate_block_structure(const spmat &A, const block_layout &layout) {
    const idx n = A.n_rows();
    const idx *row_ptr = A.row_ptr();
    const idx *col_idx = A.col_idx();
    const real *values = A.values();
    for (idx row = 0; row < n; ++row) {
        const idx block_row = layout.block_of[row];
        for (idx k = row_ptr[row]; k < row_ptr[row + 1]; ++k) {
            if (values[k] == 0.0) {
                continue;
            }
            const idx block_col = layout.block_of[col_idx[k]];
            const idx distance =
                block_row > block_col ? block_row - block_col : block_col - block_row;
            if (distance > 1) {
                throw std::invalid_argument(
                    "block_tridiagonal: entry (" + std::to_string(row) + "," +
                    std::to_string(col_idx[k]) + ") couples blocks " +
                    std::to_string(block_row) + " and " + std::to_string(block_col) +
                    ", which are not adjacent");
            }
        }
    }
}

/// @brief Dense diagonal, super- and sub-diagonal blocks in reordered indexing.
struct assembled_blocks {
    array<mat> diagonal;
    array<mat> upper;
    array<mat> lower;
};

/// @brief Scatter the sparse entries into dense blocks, scaling as it goes.
[[nodiscard]] inline assembled_blocks gather_blocks(const spmat &A, const block_layout &layout,
                                                   real scale) {
    const idx blocks = layout.offsets.size() - 1;
    assembled_blocks out;
    out.diagonal.reserve(blocks);
    for (idx k = 0; k < blocks; ++k) {
        const idx rows = layout.offsets[k + 1] - layout.offsets[k];
        out.diagonal.emplace_back(rows, rows, 0.0);
    }
    for (idx k = 0; k + 1 < blocks; ++k) {
        const idx rows = layout.offsets[k + 1] - layout.offsets[k];
        const idx cols = layout.offsets[k + 2] - layout.offsets[k + 1];
        out.upper.emplace_back(rows, cols, 0.0);
        out.lower.emplace_back(cols, rows, 0.0);
    }

    const idx n = A.n_rows();
    const idx *row_ptr = A.row_ptr();
    const idx *col_idx = A.col_idx();
    const real *values = A.values();
    for (idx row = 0; row < n; ++row) {
        const idx block_row = layout.block_of[row];
        const idx local_row = layout.position[row] - layout.offsets[block_row];
        for (idx k = row_ptr[row]; k < row_ptr[row + 1]; ++k) {
            const real value = values[k] * scale;
            if (values[k] == 0.0) {
                continue;
            }
            const idx column = col_idx[k];
            const idx block_col = layout.block_of[column];
            const idx local_col = layout.position[column] - layout.offsets[block_col];
            if (block_col == block_row) {
                out.diagonal[block_row](local_row, local_col) += value;
            } else if (block_col == block_row + 1) {
                out.upper[block_row](local_row, local_col) += value;
            } else {
                out.lower[block_col](local_row, local_col) += value;
            }
        }
    }
    return out;
}

/// @brief Permute a right-hand side into block order.
[[nodiscard]] inline mat gather_rows(const mat &B, const array<idx> &order) {
    mat out(B.rows(), B.cols(), 0.0);
    for (idx p = 0; p < B.rows(); ++p) {
        for (idx c = 0; c < B.cols(); ++c) {
            out(p, c) = B(order[p], c);
        }
    }
    return out;
}

/// @brief Undo `gather_rows`.
inline void scatter_rows(const mat &in, const array<idx> &order, mat &out) {
    for (idx p = 0; p < in.rows(); ++p) {
        for (idx c = 0; c < in.cols(); ++c) {
            out(order[p], c) = in(p, c);
        }
    }
}

[[nodiscard]] inline vec gather_rows(const vec &b, const array<idx> &order) {
    vec out(b.size(), 0.0);
    for (idx p = 0; p < b.size(); ++p) {
        out[p] = b[order[p]];
    }
    return out;
}

inline void scatter_rows(const vec &in, const array<idx> &order, vec &out) {
    for (idx p = 0; p < in.size(); ++p) {
        out[order[p]] = in[p];
    }
}

/// @brief Rows `[first, first + count)` of `M` as a dense block.
[[nodiscard]] inline mat block_rows(const mat &M, idx first, idx count) {
    mat out(count, M.cols(), 0.0);
    for (idx r = 0; r < count; ++r) {
        for (idx c = 0; c < M.cols(); ++c) {
            out(r, c) = M(first + r, c);
        }
    }
    return out;
}

inline void set_block_rows(mat &M, idx first, const mat &block) {
    for (idx r = 0; r < block.rows(); ++r) {
        for (idx c = 0; c < block.cols(); ++c) {
            M(first + r, c) = block(r, c);
        }
    }
}

[[nodiscard]] inline vec block_rows(const vec &v, idx first, idx count) {
    vec out(count, 0.0);
    for (idx r = 0; r < count; ++r) {
        out[r] = v[first + r];
    }
    return out;
}

inline void set_block_rows(vec &v, idx first, const vec &block) {
    for (idx r = 0; r < block.size(); ++r) {
        v[first + r] = block[r];
    }
}

/// @brief `C <- A B` with `C` sized here.
[[nodiscard]] inline mat product(const mat &A, const mat &B) {
    mat out(A.rows(), B.cols(), 0.0);
    matmul(A, B, out);
    return out;
}

/// @brief `y <- A x`.
[[nodiscard]] inline vec product(const mat &A, const vec &x) {
    vec out(A.rows(), 0.0);
    matvec(A, x, out);
    return out;
}

/// @brief `target <- target - update`.
inline void subtract_from(mat &target, const mat &update) {
    matadd(real(1), target, real(-1), update, target);
}

inline void subtract_from(vec &target, const vec &update) {
    for (idx i = 0; i < target.size(); ++i) {
        target[i] -= update[i];
    }
}

/// @brief Common validation and layout construction for both factorizations.
[[nodiscard]] inline block_layout prepare(const spmat &A, view<const idx> levels) {
    if (A.n_rows() != A.n_cols()) {
        throw std::invalid_argument("block_tridiagonal: matrix must be square");
    }
    if (levels.size() != A.n_rows()) {
        throw std::invalid_argument(
            "block_tridiagonal: one level is required per row, got " +
            std::to_string(levels.size()) + " for " + std::to_string(A.n_rows()) + " rows");
    }
    block_layout layout = build_block_order(levels);
    validate_block_structure(A, layout);
    return layout;
}

/// @brief Solve `L z = rhs` for lower-triangular `L`, column by column.
[[nodiscard]] inline mat forward_substitute(const mat &L, const mat &rhs) {
    const idx n = L.rows();
    mat out(n, rhs.cols(), 0.0);
    vec column(n, 0.0);
    vec solution(n, 0.0);
    for (idx c = 0; c < rhs.cols(); ++c) {
        for (idx r = 0; r < n; ++r) {
            column[r] = rhs(r, c);
        }
        kernel::trsv_lower(solution.data(), L.data(), column.data(), n);
        for (idx r = 0; r < n; ++r) {
            out(r, c) = solution[r];
        }
    }
    return out;
}

/// @brief Solve `L^T z = rhs` for lower-triangular `L`, column by column.
[[nodiscard]] inline mat backward_substitute(const mat &L, const mat &rhs) {
    const idx n = L.rows();
    mat out = rhs; // trsv_transpose_lower works in place
    vec column(n, 0.0);
    for (idx c = 0; c < rhs.cols(); ++c) {
        for (idx r = 0; r < n; ++r) {
            column[r] = rhs(r, c);
        }
        kernel::trsv_transpose_lower(column.data(), L.data(), n, n);
        for (idx r = 0; r < n; ++r) {
            out(r, c) = column[r];
        }
    }
    return out;
}

[[nodiscard]] inline vec forward_substitute(const mat &L, const vec &rhs) {
    vec out(L.rows(), 0.0);
    kernel::trsv_lower(out.data(), L.data(), rhs.data(), L.rows());
    return out;
}

[[nodiscard]] inline vec backward_substitute(const mat &L, const vec &rhs) {
    vec out = rhs;
    kernel::trsv_transpose_lower(out.data(), L.data(), L.rows(), L.rows());
    return out;
}

} // namespace detail

// =============================================================================
// Block LU
// =============================================================================

/// @brief Factor a block-tridiagonal matrix by block LU without row pivoting.
///
/// @param A Square sparse matrix whose nonzeros stay within adjacent level blocks.
/// @param levels One level label per row; arbitrary values, compressed internally.
/// @param scale Uniform scaling applied while assembling, so `factor(scale*A)`
///        needs no separate scaled copy of `A`.
/// The caller must guarantee that Gaussian elimination in the supplied block
/// ordering has nonzero pivots. Nonsingular M-matrices satisfy this condition,
/// and their diagonal Schur blocks remain nonsingular M-matrices.
///
/// @throws std::invalid_argument If `A` is not square, the level count is wrong,
///         or a nonzero couples non-adjacent blocks.
/// @throws std::runtime_error If a diagonal Schur block has a zero pivot.
[[nodiscard]] inline block_lu_factor factor_block_lu(const spmat &A,
                                                   view<const idx> levels,
                                                   real scale = 1.0) {
    const detail::block_layout layout = detail::prepare(A, levels);
    detail::assembled_blocks blocks = detail::gather_blocks(A, layout, scale);
    const idx count = layout.offsets.size() - 1;

    block_lu_factor factor;
    factor.size = A.n_rows();
    factor.offsets = layout.offsets;
    factor.order = layout.order;
    factor.upper = std::move(blocks.upper);
    factor.lower = std::move(blocks.lower);
    factor.diagonal.reserve(count);

    for (idx k = 0; k < count; ++k) {
        factor.diagonal.push_back(factor_no_pivot(assume_square(blocks.diagonal[k])));
        if (factor.diagonal.back().singular) {
            throw std::runtime_error("block_tridiagonal: zero pivot in diagonal block " +
                                     std::to_string(k));
        }
        if (k + 1 >= count) {
            break;
        }
        // lower[k] <- lower[k] D_k^{-1}. Right-side solves are not exposed
        // directly, so use Z D = L  <=>  D^T Z^T = L^T.
        mat transposed = transpose(factor.lower[k]);
        mat scaled_transposed;
        solve_transpose(factor.diagonal[k], transposed, scaled_transposed);
        factor.lower[k] = transpose(scaled_transposed);

        // D_{k+1} <- D_{k+1} - lower[k] upper[k]
        detail::subtract_from(blocks.diagonal[k + 1],
                              detail::product(factor.lower[k], factor.upper[k]));
    }
    return factor;
}

/// @brief Solve \f$AX = B\f$ using stored block LU factors.
inline void solve(const block_lu_factor &factor, const mat &B, mat &X) {
    if (B.rows() != factor.size) {
        throw std::invalid_argument("block_tridiagonal: right-hand side row count mismatch");
    }
    const idx count = factor.blocks();
    mat work = detail::gather_rows(B, factor.order);

    // Forward: Y_0 = B_0 ; Y_k = B_k - lower[k-1] Y_{k-1}
    for (idx k = 1; k < count; ++k) {
        const mat previous =
            detail::block_rows(work, factor.offsets[k - 1], factor.block_size(k - 1));
        mat current = detail::block_rows(work, factor.offsets[k], factor.block_size(k));
        detail::subtract_from(current, detail::product(factor.lower[k - 1], previous));
        detail::set_block_rows(work, factor.offsets[k], current);
    }

    // Backward: X_k = D_k^{-1} Y_k ; Y_{k-1} -= upper[k-1] X_k
    for (idx k = count; k-- > 0;) {
        const mat current = detail::block_rows(work, factor.offsets[k], factor.block_size(k));
        mat solved;
        solve(factor.diagonal[k], current, solved);
        detail::set_block_rows(work, factor.offsets[k], solved);
        if (k > 0) {
            mat previous =
                detail::block_rows(work, factor.offsets[k - 1], factor.block_size(k - 1));
            detail::subtract_from(previous, detail::product(factor.upper[k - 1], solved));
            detail::set_block_rows(work, factor.offsets[k - 1], previous);
        }
    }

    X = mat(factor.size, B.cols(), 0.0);
    detail::scatter_rows(work, factor.order, X);
}

/// @brief Solve \f$Ax = b\f$ using stored block LU factors.
inline void solve(const block_lu_factor &factor, const vec &b, vec &x) {
    if (b.size() != factor.size) {
        throw std::invalid_argument("block_tridiagonal: right-hand side size mismatch");
    }
    const idx count = factor.blocks();
    vec work = detail::gather_rows(b, factor.order);

    for (idx k = 1; k < count; ++k) {
        const vec previous =
            detail::block_rows(work, factor.offsets[k - 1], factor.block_size(k - 1));
        vec current = detail::block_rows(work, factor.offsets[k], factor.block_size(k));
        detail::subtract_from(current, detail::product(factor.lower[k - 1], previous));
        detail::set_block_rows(work, factor.offsets[k], current);
    }

    for (idx k = count; k-- > 0;) {
        const vec current = detail::block_rows(work, factor.offsets[k], factor.block_size(k));
        vec solved(current.size(), 0.0);
        solve(factor.diagonal[k], current, solved);
        detail::set_block_rows(work, factor.offsets[k], solved);
        if (k > 0) {
            vec previous =
                detail::block_rows(work, factor.offsets[k - 1], factor.block_size(k - 1));
            detail::subtract_from(previous, detail::product(factor.upper[k - 1], solved));
            detail::set_block_rows(work, factor.offsets[k - 1], previous);
        }
    }

    x = vec(factor.size, 0.0);
    detail::scatter_rows(work, factor.order, x);
}

/// @brief Solve \f$A^{T}X = B\f$ from the same factors, without refactorizing.
///
/// \f$A = \mathcal{L}\mathcal{U}\f$ gives \f$A^{T} = \mathcal{U}^{T}\mathcal{L}^{T}\f$,
/// so the block dependency runs the other way: a forward sweep through
/// \f$\mathcal{U}^{T}\f$ (block lower bidiagonal, `upper[k]^T` below the
/// diagonal) followed by a backward sweep through \f$\mathcal{L}^{T}\f$ (block
/// unit upper bidiagonal, `lower[k]^T` above it).
inline void solve_transpose(const block_lu_factor &factor, const mat &B, mat &X) {
    if (B.rows() != factor.size) {
        throw std::invalid_argument("block_tridiagonal: right-hand side row count mismatch");
    }
    const idx count = factor.blocks();
    mat work = detail::gather_rows(B, factor.order);

    // U^T Z = B : Z_k = D_k^{-T}(B_k - upper[k-1]^T Z_{k-1})
    for (idx k = 0; k < count; ++k) {
        mat current = detail::block_rows(work, factor.offsets[k], factor.block_size(k));
        if (k > 0) {
            const mat previous =
                detail::block_rows(work, factor.offsets[k - 1], factor.block_size(k - 1));
            detail::subtract_from(current,
                                  detail::product(transpose(factor.upper[k - 1]), previous));
        }
        mat solved;
        solve_transpose(factor.diagonal[k], current, solved);
        detail::set_block_rows(work, factor.offsets[k], solved);
    }

    // L^T X = Z : X_k = Z_k - lower[k]^T X_{k+1}
    for (idx k = count - 1; k-- > 0;) {
        const mat next =
            detail::block_rows(work, factor.offsets[k + 1], factor.block_size(k + 1));
        mat current = detail::block_rows(work, factor.offsets[k], factor.block_size(k));
        detail::subtract_from(current, detail::product(transpose(factor.lower[k]), next));
        detail::set_block_rows(work, factor.offsets[k], current);
    }

    X = mat(factor.size, B.cols(), 0.0);
    detail::scatter_rows(work, factor.order, X);
}

/// @brief Solve \f$A^{T}x = b\f$ from the same factors, without refactorizing.
inline void solve_transpose(const block_lu_factor &factor, const vec &b, vec &x) {
    if (b.size() != factor.size) {
        throw std::invalid_argument("block_tridiagonal: right-hand side size mismatch");
    }
    const idx count = factor.blocks();
    vec work = detail::gather_rows(b, factor.order);

    for (idx k = 0; k < count; ++k) {
        vec current = detail::block_rows(work, factor.offsets[k], factor.block_size(k));
        if (k > 0) {
            const vec previous =
                detail::block_rows(work, factor.offsets[k - 1], factor.block_size(k - 1));
            detail::subtract_from(current,
                                  detail::product(transpose(factor.upper[k - 1]), previous));
        }
        vec solved(current.size(), 0.0);
        solve_transpose(factor.diagonal[k], current, solved);
        detail::set_block_rows(work, factor.offsets[k], solved);
    }

    for (idx k = count - 1; k-- > 0;) {
        const vec next =
            detail::block_rows(work, factor.offsets[k + 1], factor.block_size(k + 1));
        vec current = detail::block_rows(work, factor.offsets[k], factor.block_size(k));
        detail::subtract_from(current, detail::product(transpose(factor.lower[k]), next));
        detail::set_block_rows(work, factor.offsets[k], current);
    }

    x = vec(factor.size, 0.0);
    detail::scatter_rows(work, factor.order, x);
}

// =============================================================================
// Block Cholesky
// =============================================================================

/// @brief Factor a symmetric positive-definite block-tridiagonal matrix.
///
/// Only the diagonal and sub-diagonal blocks are used; the super-diagonal blocks
/// are taken to be their transposes, so an input storing both halves and one
/// storing only the lower half factor identically.
///
/// @throws std::invalid_argument If `A` is not square, the level count is wrong,
///         or a nonzero couples non-adjacent blocks.
/// @throws std::runtime_error If a diagonal block is not positive definite.
[[nodiscard]] inline block_cholesky_factor factor_block_cholesky(const spmat &A,
                                                               view<const idx> levels) {
    const detail::block_layout layout = detail::prepare(A, levels);
    detail::assembled_blocks blocks = detail::gather_blocks(A, layout, real(1));
    const idx count = layout.offsets.size() - 1;

    block_cholesky_factor factor;
    factor.size = A.n_rows();
    factor.offsets = layout.offsets;
    factor.order = layout.order;
    factor.lower = std::move(blocks.lower);
    factor.diagonal.reserve(count);

    for (idx k = 0; k < count; ++k) {
        auto result = cholesky(assume_spd(blocks.diagonal[k]));
        if (!result.success) {
            throw std::runtime_error("block_tridiagonal: diagonal block " + std::to_string(k) +
                                     " is not positive definite");
        }
        factor.diagonal.push_back(std::move(result));
        if (k + 1 >= count) {
            break;
        }
        // C <- C L_k^{-T}, i.e. L_k Z^T = C^T: forward-substitute each row of C.
        const mat &chol = factor.diagonal[k].L;
        factor.lower[k] = transpose(detail::forward_substitute(chol, transpose(factor.lower[k])));

        // A_{k+1} <- A_{k+1} - C C^T
        detail::subtract_from(blocks.diagonal[k + 1],
                              detail::product(factor.lower[k], transpose(factor.lower[k])));
    }
    return factor;
}

/// @brief Solve \f$AX = B\f$ using stored block Cholesky factors.
inline void solve(const block_cholesky_factor &factor, const mat &B, mat &X) {
    if (B.rows() != factor.size) {
        throw std::invalid_argument("block_tridiagonal: right-hand side row count mismatch");
    }
    const idx count = factor.blocks();
    mat work = detail::gather_rows(B, factor.order);

    // L Y = B
    for (idx k = 0; k < count; ++k) {
        mat current = detail::block_rows(work, factor.offsets[k], factor.block_size(k));
        if (k > 0) {
            const mat previous =
                detail::block_rows(work, factor.offsets[k - 1], factor.block_size(k - 1));
            detail::subtract_from(current, detail::product(factor.lower[k - 1], previous));
        }
        detail::set_block_rows(work, factor.offsets[k],
                               detail::forward_substitute(factor.diagonal[k].L, current));
    }

    // L^T X = Y
    for (idx k = count; k-- > 0;) {
        mat current = detail::block_rows(work, factor.offsets[k], factor.block_size(k));
        if (k + 1 < count) {
            const mat next =
                detail::block_rows(work, factor.offsets[k + 1], factor.block_size(k + 1));
            detail::subtract_from(current, detail::product(transpose(factor.lower[k]), next));
        }
        detail::set_block_rows(work, factor.offsets[k],
                               detail::backward_substitute(factor.diagonal[k].L, current));
    }

    X = mat(factor.size, B.cols(), 0.0);
    detail::scatter_rows(work, factor.order, X);
}

/// @brief Solve \f$Ax = b\f$ using stored block Cholesky factors.
inline void solve(const block_cholesky_factor &factor, const vec &b, vec &x) {
    if (b.size() != factor.size) {
        throw std::invalid_argument("block_tridiagonal: right-hand side size mismatch");
    }
    const idx count = factor.blocks();
    vec work = detail::gather_rows(b, factor.order);

    for (idx k = 0; k < count; ++k) {
        vec current = detail::block_rows(work, factor.offsets[k], factor.block_size(k));
        if (k > 0) {
            const vec previous =
                detail::block_rows(work, factor.offsets[k - 1], factor.block_size(k - 1));
            detail::subtract_from(current, detail::product(factor.lower[k - 1], previous));
        }
        detail::set_block_rows(work, factor.offsets[k],
                               detail::forward_substitute(factor.diagonal[k].L, current));
    }

    for (idx k = count; k-- > 0;) {
        vec current = detail::block_rows(work, factor.offsets[k], factor.block_size(k));
        if (k + 1 < count) {
            const vec next =
                detail::block_rows(work, factor.offsets[k + 1], factor.block_size(k + 1));
            detail::subtract_from(current, detail::product(transpose(factor.lower[k]), next));
        }
        detail::set_block_rows(work, factor.offsets[k],
                               detail::backward_substitute(factor.diagonal[k].L, current));
    }

    x = vec(factor.size, 0.0);
    detail::scatter_rows(work, factor.order, x);
}

} // namespace num
