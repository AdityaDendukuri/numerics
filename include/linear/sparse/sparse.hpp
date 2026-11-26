/// @file linear/sparse/sparse.hpp
/// @brief Sparse matrices in compressed sparse row (CSR) format, and operations on them.
#pragma once
#include "container/concepts.hpp"
#include "container/matrix.hpp"
#include "container/vector.hpp"
#include "core/types.hpp"
#include "kernel/kernel.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <span>
#include <stdexcept>
#include <vector>

namespace num {

/// @brief Sparse matrix in compressed sparse row (CSR) format.
/// Row `i` holds its values and column indices in positions `[row_ptr[i], row_ptr[i+1])`.
class spmat {
  public:
    /// @brief Construct a matrix from CSR arrays, taking ownership of them.
    spmat(idx n_rows, idx n_cols, array<real> vals, array<idx> col_idx, array<idx> row_ptr);

    /// @brief Construct a matrix from (row, column, value) triplets.
    /// Duplicate entries are summed; the triplets need not be sorted.
    static spmat from_triplets(idx n_rows, idx n_cols, const array<idx> &rows,
                               const array<idx> &cols, const array<real> &vals);

    /// @brief Construct a matrix from zero-based compressed column (CSC) arrays.
    /// Only the first `col_ptrs.back()` entries of the value and index arrays are read.
    static spmat from_csc(idx n_rows, idx n_cols, const array<real> &vals,
                          const array<idx> &row_indices, const array<idx> &col_ptrs);

    [[nodiscard]] idx n_rows() const { return n_rows_; }
    [[nodiscard]] idx n_cols() const { return n_cols_; }
    [[nodiscard]] idx rows() const noexcept { return n_rows_; }
    [[nodiscard]] idx cols() const noexcept { return n_cols_; }
    [[nodiscard]] idx nnz() const { return vals_.size(); }

    /// @brief Return the entry \f$a_{ij}\f$, or 0 if it is not stored. Costs
    /// \f$O(\mathrm{nnz}/n)\f$.
    real operator()(idx i, idx j) const;

    [[nodiscard]] const real *values() const { return vals_.data(); }
    [[nodiscard]] const idx *col_idx() const { return col_idx_.data(); }
    [[nodiscard]] const idx *row_ptr() const { return row_ptr_.data(); }

    /// @brief Apply the matrix: \f$y \leftarrow A x\f$.
    template <class X = vec<real>, class Y = vec<real>>
    void apply(const X &x, Y &y) const {
        if constexpr (std::is_same_v<X, vec<real>> && std::is_same_v<Y, vec<real>>) {
            sparse_matvec(*this, x, y);
        } else {
            for (idx i = 0; i < n_rows_; ++i) {
                real sum = 0.0;
                for (idx k = row_ptr_[i]; k < row_ptr_[i + 1]; ++k) {
                    sum += vals_[k] * x[col_idx_[k]];
                }
                y[i] = sum;
            }
        }
    }

  private:
    idx n_rows_ = 0, n_cols_ = 0;
    array<real> vals_;
    array<idx> col_idx_;
    array<idx> row_ptr_; // size n_rows_ + 1
};

/// @brief Multiply a sparse matrix and a vector: \f$y \leftarrow A x\f$, in \f$O(\mathrm{nnz})\f$.
/// @throws std::invalid_argument If the dimensions do not agree.
void sparse_matvec(const spmat &A, const vec<real> &x, vec<real> &y);

/// @brief Scale the rows and columns of a sparse matrix, returning a dense one:
/// \f$D^{-1} A D\f$, \f$D = \mathrm{diag}(w)\f$.
/// @throws std::invalid_argument If the dimensions disagree or a weight is not positive.
[[nodiscard]] mat<real> diagonal_similarity(const spmat &A, view<const real> weights);

/// @brief Scale the rows and columns of a sparse matrix, keeping it sparse: \f$D^{-1} A D\f$.
/// The result has the pattern of `A`.
/// @throws std::invalid_argument If the dimensions disagree or a weight is not positive.
[[nodiscard]] spmat sparse_diagonal_similarity(const spmat &A, view<const real> weights);

/// @brief Compute the symmetric part of a square matrix: \f$(A + A^T) / 2\f$.
/// @throws std::invalid_argument If `A` is not square.
[[nodiscard]] spmat symmetric_part(const spmat &A);

/// @brief Scale the rows and columns of a sparse matrix by the same weights:
/// \f$W A W\f$, \f$W = \mathrm{diag}(w)\f$.
/// The result has the pattern of `A`.
/// @throws std::invalid_argument If the dimensions disagree.
[[nodiscard]] spmat sparse_congruence(const spmat &A, view<const real> weights);

inline spmat::spmat(idx n_rows, idx n_cols, array<real> vals, array<idx> col_idx,
                    array<idx> row_ptr)
    : n_rows_(n_rows), n_cols_(n_cols), vals_(std::move(vals)), col_idx_(std::move(col_idx)),
      row_ptr_(std::move(row_ptr)) {
    if (row_ptr_.size() != n_rows_ + 1) {
        throw std::invalid_argument("spmat: row_ptr must have length n_rows+1");
    }
    if (col_idx_.size() != vals_.size()) {
        throw std::invalid_argument("spmat: col_idx and vals must have equal length");
    }
}

inline spmat spmat::from_triplets(idx n_rows, idx n_cols, const array<idx> &rows,
                                  const array<idx> &cols, const array<real> &vals) {
    if (rows.size() != cols.size() || rows.size() != vals.size()) {
        throw std::invalid_argument("spmat::from_triplets: inconsistent input sizes");
    }

    // Count entries per row
    array<idx> row_count(n_rows, 0);
    for (idx k = 0; k < rows.size(); ++k) {
        if (rows[k] >= n_rows || cols[k] >= n_cols) {
            throw std::out_of_range("spmat::from_triplets: index out of range");
        }
        ++row_count[rows[k]];
    }

    // Build row_ptr
    array<idx> row_ptr(n_rows + 1, 0);
    for (idx i = 0; i < n_rows; ++i) {
        row_ptr[i + 1] = row_ptr[i] + row_count[i];
    }

    idx nnz = row_ptr[n_rows];
    array<real> out_vals(nnz, 0.0);
    array<idx> out_col(nnz);

    // Fill entries (stable insertion within each row)
    array<idx> fill_pos = row_ptr;
    for (idx k = 0; k < rows.size(); ++k) {
        idx pos = fill_pos[rows[k]]++;
        out_col[pos] = cols[k];
        out_vals[pos] = vals[k];
    }

    // Sort each row by column and sum duplicates
    for (idx i = 0; i < n_rows; ++i) {
        idx start = row_ptr[i], end = row_ptr[i + 1];
        // Sort by column index
        array<idx> order(end - start);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(),
                  [&](idx a, idx b) { return out_col[start + a] < out_col[start + b]; });

        array<real> sv(end - start);
        array<idx> sc(end - start);
        for (idx k = 0; k < order.size(); ++k) {
            sv[k] = out_vals[start + order[k]];
            sc[k] = out_col[start + order[k]];
        }
        for (idx k = 0; k < order.size(); ++k) {
            out_vals[start + k] = sv[k];
            out_col[start + k] = sc[k];
        }

        // Sum duplicates in-place
        idx write = start;
        for (idx k = start; k < end;) {
            idx cur_col = out_col[k];
            real sum = 0.0;
            while (k < end && out_col[k] == cur_col) {
                sum += out_vals[k++];
            }
            out_col[write] = cur_col;
            out_vals[write++] = sum;
        }
        // Compact row_ptr if duplicates were merged
        row_ptr[i + 1] = write;
        // Shift remaining rows' data (rare; only matters if duplicates exist)
        if (write < end) {
            for (idx k = end; k < nnz; ++k) {
                out_vals[write + (k - end)] = out_vals[k];
                out_col[write + (k - end)] = out_col[k];
            }
            nnz -= (end - write);
            out_vals.resize(nnz);
            out_col.resize(nnz);
            // Fix subsequent row_ptr entries
            idx delta = end - write;
            for (idx r = i + 2; r <= n_rows; ++r) {
                row_ptr[r] -= delta;
            }
        }
    }

    return spmat(n_rows, n_cols, std::move(out_vals), std::move(out_col), std::move(row_ptr));
}

inline spmat spmat::from_csc(idx n_rows, idx n_cols, const array<real> &vals,
                             const array<idx> &row_indices, const array<idx> &col_ptrs) {
    if (col_ptrs.size() != n_cols + 1) {
        throw std::invalid_argument("spmat::from_csc: col_ptrs must have length n_cols+1");
    }
    if (col_ptrs.empty() || col_ptrs.front() != 0) {
        throw std::invalid_argument("spmat::from_csc: col_ptrs must start at zero");
    }
    for (idx j = 0; j < n_cols; ++j) {
        if (col_ptrs[j] > col_ptrs[j + 1]) {
            throw std::invalid_argument("spmat::from_csc: col_ptrs must be nondecreasing");
        }
    }

    // Armadillo's serializer historically emitted n_nonzero+1 payload
    // entries while the final column pointer still reports n_nonzero.  Use
    // the pointer, rather than payload length, as the authoritative nnz.
    const idx nnz = col_ptrs.back();
    if (nnz > vals.size() || nnz > row_indices.size()) {
        throw std::invalid_argument("spmat::from_csc: payload shorter than col_ptrs.back()");
    }

    array<idx> row_ptr(n_rows + 1, 0);
    for (idx k = 0; k < nnz; ++k) {
        if (row_indices[k] >= n_rows) {
            throw std::out_of_range("spmat::from_csc: row index out of range");
        }
        ++row_ptr[row_indices[k] + 1];
    }
    for (idx i = 0; i < n_rows; ++i) {
        row_ptr[i + 1] += row_ptr[i];
    }

    array<real> out_vals(nnz);
    array<idx> out_col(nnz);
    array<idx> next = row_ptr;
    for (idx j = 0; j < n_cols; ++j) {
        for (idx k = col_ptrs[j]; k < col_ptrs[j + 1]; ++k) {
            const idx i = row_indices[k];
            const idx p = next[i]++;
            out_vals[p] = vals[k];
            out_col[p] = j;
        }
    }

    return spmat(n_rows, n_cols, std::move(out_vals), std::move(out_col), std::move(row_ptr));
}

inline real spmat::operator()(idx i, idx j) const {
    for (idx k = row_ptr_[i]; k < row_ptr_[i + 1]; ++k) {
        if (col_idx_[k] == j) {
            return vals_[k];
        }
    }
    return 0.0;
}

/// @brief Multiply any CSR sparse matrix and a vector: \f$y \leftarrow A x\f$.
/// Accepts any type with the CSR accessors, not only `spmat`.
template <class M>
requires repr::csr<M> inline void matvec(const M &A, const vec<real> &x, vec<real> &y) {
    if (A.n_cols() != x.size() || A.n_rows() != y.size()) {
        throw std::invalid_argument("Dimension mismatch in matvec");
    }
    kernel::spmv(y.data(), A.values(), A.row_ptr(), A.col_idx(), x.data(), A.n_rows());
}

/// @brief Multiply a sparse matrix and a vector:
/// \f$y \leftarrow A x\f$. Prefer the general `matvec`.
inline void sparse_matvec(const spmat &A, const vec<real> &x, vec<real> &y) {
    matvec(A, x, y);
}

/// @brief Scale a sparse matrix, keeping its pattern: \f$\alpha A\f$.
template <class M>
requires repr::csr<M> inline spmat scaled(const M &A, real alpha) {
    array<real> values(A.values(), A.values() + A.nnz());
    for (real &value : values) {
        value *= alpha;
    }
    return {A.n_rows(), A.n_cols(), std::move(values),
            array<idx>(A.col_idx(), A.col_idx() + A.nnz()),
            array<idx>(A.row_ptr(), A.row_ptr() + A.n_rows() + 1)};
}

/// @brief Transpose a sparse matrix, in \f$O(\mathrm{nnz} + n)\f$: \f$A^T\f$.
template <class M>
requires repr::csr<M> inline spmat transpose(const M &A) {
    array<idx> column_ptr(A.n_cols() + 1, 0);
    for (idx entry = 0; entry < A.nnz(); ++entry) {
        ++column_ptr[A.col_idx()[entry] + 1];
    }
    for (idx column = 0; column < A.n_cols(); ++column) {
        column_ptr[column + 1] += column_ptr[column];
    }

    array<real> values(A.nnz());
    array<idx> columns(A.nnz());
    array<idx> next = column_ptr;
    for (idx row = 0; row < A.n_rows(); ++row) {
        for (idx entry = A.row_ptr()[row]; entry < A.row_ptr()[row + 1]; ++entry) {
            const idx destination = next[A.col_idx()[entry]]++;
            values[destination] = A.values()[entry];
            columns[destination] = row;
        }
    }
    return {A.n_cols(), A.n_rows(), std::move(values), std::move(columns), std::move(column_ptr)};
}

/// @brief Convert a CSR matrix to a dense one.
template <class M>
requires repr::csr<M> inline mat<real> dense(const M &A) {
    mat<real> result(A.n_rows(), A.n_cols(), 0.0);
    for (idx row = 0; row < A.n_rows(); ++row) {
        for (idx entry = A.row_ptr()[row]; entry < A.row_ptr()[row + 1]; ++entry) {
            result(row, A.col_idx()[entry]) = A.values()[entry];
        }
    }
    return result;
}

/// @brief Return the main diagonal of a sparse matrix, with 0 for entries not stored:
/// \f$d_i = a_{ii}\f$.
template <class M>
requires repr::csr<M> inline vec<real> diagonal(const M &A) {
    const idx n = std::min(A.n_rows(), A.n_cols());
    vec<real> result(n, 0.0);
    for (idx row = 0; row < n; ++row) {
        for (idx entry = A.row_ptr()[row]; entry < A.row_ptr()[row + 1]; ++entry) {
            if (A.col_idx()[entry] == row) {
                result[row] = A.values()[entry];
                break;
            }
        }
    }
    return result;
}

inline spmat symmetric_part(const spmat &A) {
    if (A.n_rows() != A.n_cols()) {
        throw std::invalid_argument("symmetric_part: matrix must be square");
    }
    array<idx> rows, columns;
    array<real> values;
    for (idx row = 0; row < A.n_rows(); ++row) {
        for (idx entry = A.row_ptr()[row]; entry < A.row_ptr()[row + 1]; ++entry) {
            const idx column = A.col_idx()[entry];
            const real half = 0.5 * A.values()[entry];
            rows.push_back(row);
            columns.push_back(column);
            values.push_back(half);
            rows.push_back(column);
            columns.push_back(row);
            values.push_back(half);
        }
    }
    return spmat::from_triplets(A.n_rows(), A.n_cols(), rows, columns, values);
}

inline spmat sparse_congruence(const spmat &A, view<const real> weights) {
    if (A.n_rows() != A.n_cols() || weights.size() != A.n_rows()) {
        throw std::invalid_argument("sparse_congruence: dimensions must match");
    }
    array<real> values(A.values(), A.values() + A.nnz());
    for (idx row = 0; row < A.n_rows(); ++row) {
        for (idx entry = A.row_ptr()[row]; entry < A.row_ptr()[row + 1]; ++entry) {
            values[entry] *= weights[row] * weights[A.col_idx()[entry]];
        }
    }
    return {A.n_rows(), A.n_cols(), std::move(values),
            array<idx>(A.col_idx(), A.col_idx() + A.nnz()),
            array<idx>(A.row_ptr(), A.row_ptr() + A.n_rows() + 1)};
}

inline spmat sparse_diagonal_similarity(const spmat &A, view<const real> weights) {
    if (A.n_rows() != A.n_cols() || weights.size() != A.n_rows()) {
        throw std::invalid_argument("sparse_diagonal_similarity: dimensions must match");
    }
    if (!std::all_of(weights.begin(), weights.end(), [](real value) { return value > 0.0; })) {
        throw std::invalid_argument("sparse_diagonal_similarity: weights must be positive");
    }
    array<real> values(A.values(), A.values() + A.nnz());
    for (idx row = 0; row < A.n_rows(); ++row) {
        for (idx entry = A.row_ptr()[row]; entry < A.row_ptr()[row + 1]; ++entry) {
            values[entry] *= weights[A.col_idx()[entry]] / weights[row];
        }
    }
    return {A.n_rows(), A.n_cols(), std::move(values),
            array<idx>(A.col_idx(), A.col_idx() + A.nnz()),
            array<idx>(A.row_ptr(), A.row_ptr() + A.n_rows() + 1)};
}

inline mat<real> diagonal_similarity(const spmat &A, view<const real> weights) {
    if (A.n_rows() != A.n_cols() || weights.size() != A.n_rows()) {
        throw std::invalid_argument("diagonal_similarity: dimensions must match");
    }
    if (!std::all_of(weights.begin(), weights.end(), [](real value) { return value > 0.0; })) {
        throw std::invalid_argument("diagonal_similarity: weights must be positive");
    }
    mat<real> result(A.n_rows(), A.n_cols(), 0.0);
    for (idx row = 0; row < A.n_rows(); ++row) {
        for (idx entry = A.row_ptr()[row]; entry < A.row_ptr()[row + 1]; ++entry) {
            const idx column = A.col_idx()[entry];
            result(row, column) = A.values()[entry] * weights[column] / weights[row];
        }
    }
    return result;
}

/// @brief Return A with every diagonal entry stored, adding zeros where absent.
[[nodiscard]] inline spmat with_explicit_diagonal(const spmat &A) {
    array<idx> rows, cols;
    array<real> vals;
    for (idx i = 0; i < A.n_rows(); ++i) {
        bool has_diagonal = false;
        for (idx k = A.row_ptr()[i]; k < A.row_ptr()[i + 1]; ++k) {
            rows.push_back(i);
            cols.push_back(A.col_idx()[k]);
            vals.push_back(A.values()[k]);
            has_diagonal = has_diagonal || A.col_idx()[k] == i;
        }
        if (!has_diagonal && i < A.n_cols()) {
            rows.push_back(i);
            cols.push_back(i);
            vals.push_back(0.0);
        }
    }
    return spmat::from_triplets(A.n_rows(), A.n_cols(), rows, cols, vals);
}

/// @brief Construct \f$sI - A\f$ on the sparsity pattern of A plus the diagonal.
[[nodiscard]] inline spmat characteristic_matrix(const spmat &A, real s) {
    const spmat D = with_explicit_diagonal(A);
    array<real> vals(D.nnz());
    array<idx> cols(D.col_idx(), D.col_idx() + D.nnz()),
        ptr(D.row_ptr(), D.row_ptr() + D.n_rows() + 1);
    for (idx i = 0; i < D.n_rows(); ++i) {
        for (idx k = ptr[i]; k < ptr[i + 1]; ++k) {
            vals[k] = (cols[k] == i ? s : 0.0) - D.values()[k];
        }
    }
    return spmat(D.n_rows(), D.n_cols(), std::move(vals), std::move(cols), std::move(ptr));
}

/// @brief Form \f$\sum_k w_k T_k\f$ for sparse matrices of equal size, on the union of their
/// patterns.
[[nodiscard]] inline spmat sparse_linear_combination(const std::vector<spmat> &terms,
                                                     const std::vector<real> &weights) {
    if (terms.empty() || terms.size() != weights.size()) {
        throw std::invalid_argument("sparse_linear_combination: need one weight per term");
    }
    array<idx> rows, cols;
    array<real> vals;
    for (idx t = 0; t < terms.size(); ++t) {
        const spmat &T = terms[t];
        if (T.n_rows() != terms[0].n_rows() || T.n_cols() != terms[0].n_cols()) {
            throw std::invalid_argument("sparse_linear_combination: size mismatch");
        }
        for (idx i = 0; i < T.n_rows(); ++i) {
            for (idx k = T.row_ptr()[i]; k < T.row_ptr()[i + 1]; ++k) {
                rows.push_back(i);
                cols.push_back(T.col_idx()[k]);
                vals.push_back(weights[t] * T.values()[k]);
            }
        }
    }
    return spmat::from_triplets(terms[0].n_rows(), terms[0].n_cols(), rows, cols, vals);
}

} // namespace num
