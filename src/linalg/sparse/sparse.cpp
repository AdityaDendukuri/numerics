#include "linalg/sparse/sparse.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace num {

SparseMatrix::SparseMatrix(idx n_rows, idx n_cols, std::vector<real> vals, std::vector<idx> col_idx,
                           std::vector<idx> row_ptr)
    : n_rows_(n_rows), n_cols_(n_cols), vals_(std::move(vals)), col_idx_(std::move(col_idx)),
      row_ptr_(std::move(row_ptr)) {
    if (row_ptr_.size() != n_rows_ + 1) {
        throw std::invalid_argument("SparseMatrix: row_ptr must have length n_rows+1");
    }
    if (col_idx_.size() != vals_.size()) {
        throw std::invalid_argument("SparseMatrix: col_idx and vals must have equal length");
    }
}

SparseMatrix SparseMatrix::from_triplets(idx n_rows, idx n_cols, const std::vector<idx> &rows,
                                         const std::vector<idx> &cols,
                                         const std::vector<real> &vals) {
    if (rows.size() != cols.size() || rows.size() != vals.size()) {
        throw std::invalid_argument("SparseMatrix::from_triplets: inconsistent input sizes");
    }

    // Count entries per row
    std::vector<idx> row_count(n_rows, 0);
    for (idx k = 0; k < rows.size(); ++k) {
        if (rows[k] >= n_rows || cols[k] >= n_cols) {
            throw std::out_of_range("SparseMatrix::from_triplets: index out of range");
        }
        ++row_count[rows[k]];
    }

    // Build row_ptr
    std::vector<idx> row_ptr(n_rows + 1, 0);
    for (idx i = 0; i < n_rows; ++i) {
        row_ptr[i + 1] = row_ptr[i] + row_count[i];
    }

    idx nnz = row_ptr[n_rows];
    std::vector<real> out_vals(nnz, 0.0);
    std::vector<idx> out_col(nnz);

    // Fill entries (stable insertion within each row)
    std::vector<idx> fill_pos = row_ptr;
    for (idx k = 0; k < rows.size(); ++k) {
        idx pos = fill_pos[rows[k]]++;
        out_col[pos] = cols[k];
        out_vals[pos] = vals[k];
    }

    // Sort each row by column and sum duplicates
    for (idx i = 0; i < n_rows; ++i) {
        idx start = row_ptr[i], end = row_ptr[i + 1];
        // Sort by column index
        std::vector<idx> order(end - start);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(),
                  [&](idx a, idx b) { return out_col[start + a] < out_col[start + b]; });

        std::vector<real> sv(end - start);
        std::vector<idx> sc(end - start);
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

    return SparseMatrix(n_rows, n_cols, std::move(out_vals), std::move(out_col),
                        std::move(row_ptr));
}

SparseMatrix SparseMatrix::from_csc(idx n_rows, idx n_cols, const std::vector<real> &vals,
                                    const std::vector<idx> &row_indices,
                                    const std::vector<idx> &col_ptrs) {
    if (col_ptrs.size() != n_cols + 1) {
        throw std::invalid_argument("SparseMatrix::from_csc: col_ptrs must have length n_cols+1");
    }
    if (col_ptrs.empty() || col_ptrs.front() != 0) {
        throw std::invalid_argument("SparseMatrix::from_csc: col_ptrs must start at zero");
    }
    for (idx j = 0; j < n_cols; ++j) {
        if (col_ptrs[j] > col_ptrs[j + 1]) {
            throw std::invalid_argument("SparseMatrix::from_csc: col_ptrs must be nondecreasing");
        }
    }

    // Armadillo's serializer historically emitted n_nonzero+1 payload
    // entries while the final column pointer still reports n_nonzero.  Use
    // the pointer, rather than payload length, as the authoritative nnz.
    const idx nnz = col_ptrs.back();
    if (nnz > vals.size() || nnz > row_indices.size()) {
        throw std::invalid_argument("SparseMatrix::from_csc: payload shorter than col_ptrs.back()");
    }

    std::vector<idx> row_ptr(n_rows + 1, 0);
    for (idx k = 0; k < nnz; ++k) {
        if (row_indices[k] >= n_rows) {
            throw std::out_of_range("SparseMatrix::from_csc: row index out of range");
        }
        ++row_ptr[row_indices[k] + 1];
    }
    for (idx i = 0; i < n_rows; ++i) {
        row_ptr[i + 1] += row_ptr[i];
    }

    std::vector<real> out_vals(nnz);
    std::vector<idx> out_col(nnz);
    std::vector<idx> next = row_ptr;
    for (idx j = 0; j < n_cols; ++j) {
        for (idx k = col_ptrs[j]; k < col_ptrs[j + 1]; ++k) {
            const idx i = row_indices[k];
            const idx p = next[i]++;
            out_vals[p] = vals[k];
            out_col[p] = j;
        }
    }

    return SparseMatrix(n_rows, n_cols, std::move(out_vals), std::move(out_col),
                        std::move(row_ptr));
}

real SparseMatrix::operator()(idx i, idx j) const {
    for (idx k = row_ptr_[i]; k < row_ptr_[i + 1]; ++k) {
        if (col_idx_[k] == j) {
            return vals_[k];
        }
    }
    return 0.0;
}

void sparse_matvec(const SparseMatrix &A, const Vector &x, Vector &y) {
    if (A.n_cols() != x.size() || A.n_rows() != y.size()) {
        throw std::invalid_argument("Dimension mismatch in sparse_matvec");
    }
    for (idx i = 0; i < A.n_rows(); ++i) {
        real sum = 0.0;
        for (idx k = A.row_ptr()[i]; k < A.row_ptr()[i + 1]; ++k) {
            sum += A.values()[k] * x[A.col_idx()[k]];
        }
        y[i] = sum;
    }
}

SparseMatrix scaled(const SparseMatrix &A, real alpha) {
    std::vector<real> values(A.values(), A.values() + A.nnz());
    for (real &value : values) {
        value *= alpha;
    }
    return {A.n_rows(), A.n_cols(), std::move(values),
            std::vector<idx>(A.col_idx(), A.col_idx() + A.nnz()),
            std::vector<idx>(A.row_ptr(), A.row_ptr() + A.n_rows() + 1)};
}

SparseMatrix transpose(const SparseMatrix &A) {
    std::vector<idx> column_ptr(A.n_cols() + 1, 0);
    for (idx entry = 0; entry < A.nnz(); ++entry) {
        ++column_ptr[A.col_idx()[entry] + 1];
    }
    for (idx column = 0; column < A.n_cols(); ++column) {
        column_ptr[column + 1] += column_ptr[column];
    }

    std::vector<real> values(A.nnz());
    std::vector<idx> columns(A.nnz());
    std::vector<idx> next = column_ptr;
    for (idx row = 0; row < A.n_rows(); ++row) {
        for (idx entry = A.row_ptr()[row]; entry < A.row_ptr()[row + 1]; ++entry) {
            const idx destination = next[A.col_idx()[entry]]++;
            values[destination] = A.values()[entry];
            columns[destination] = row;
        }
    }
    return {A.n_cols(), A.n_rows(), std::move(values), std::move(columns), std::move(column_ptr)};
}

Matrix dense(const SparseMatrix &A) {
    Matrix result(A.n_rows(), A.n_cols(), 0.0);
    for (idx row = 0; row < A.n_rows(); ++row) {
        for (idx entry = A.row_ptr()[row]; entry < A.row_ptr()[row + 1]; ++entry) {
            result(row, A.col_idx()[entry]) = A.values()[entry];
        }
    }
    return result;
}

Vector diagonal(const SparseMatrix &A) {
    const idx n = std::min(A.n_rows(), A.n_cols());
    Vector result(n, 0.0);
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

Matrix diagonal_similarity(const SparseMatrix &A, std::span<const real> weights) {
    if (A.n_rows() != A.n_cols() || weights.size() != A.n_rows()) {
        throw std::invalid_argument("diagonal_similarity: dimensions must match");
    }
    if (!std::all_of(weights.begin(), weights.end(), [](real value) { return value > 0.0; })) {
        throw std::invalid_argument("diagonal_similarity: weights must be positive");
    }
    Matrix result(A.n_rows(), A.n_cols(), 0.0);
    for (idx row = 0; row < A.n_rows(); ++row) {
        for (idx entry = A.row_ptr()[row]; entry < A.row_ptr()[row + 1]; ++entry) {
            const idx column = A.col_idx()[entry];
            result(row, column) = A.values()[entry] * weights[column] / weights[row];
        }
    }
    return result;
}

} // namespace num
