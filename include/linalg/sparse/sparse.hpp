/// @file sparse.hpp
/// @brief Compressed Sparse Row (CSR) matrix and operations
#pragma once
#include "core/types.hpp"
#include "core/vector.hpp"
#include <vector>
#include <stdexcept>

namespace num {

/// @brief Sparse matrix in Compressed Sparse Row (CSR) format
///
/// Non-zero values for row i are stored in vals_[row_ptr_[i] .. row_ptr_[i+1]).
/// Corresponding column indices are in col_idx_[row_ptr_[i] .. row_ptr_[i+1]).
class SparseMatrix {
public:
    /// @brief Construct from raw CSR arrays (takes ownership)
    SparseMatrix(idx n_rows, idx n_cols,
                 std::vector<real> vals,
                 std::vector<idx>  col_idx,
                 std::vector<idx>  row_ptr);

    /// @brief Build from coordinate (COO / triplet) lists
    ///
    /// Duplicate (row, col) entries are summed. Entries need not be sorted.
    static SparseMatrix from_triplets(idx n_rows, idx n_cols,
                                      const std::vector<idx>&  rows,
                                      const std::vector<idx>&  cols,
                                      const std::vector<real>& vals);

    idx  n_rows() const { return n_rows_; }
    idx  n_cols() const { return n_cols_; }
    idx  nnz()    const { return vals_.size(); }

    /// @brief Element access A(i,j); returns 0 if outside stored pattern  -- O(nnz/n)
    real operator()(idx i, idx j) const;

    const real* values()  const { return vals_.data(); }
    const idx*  col_idx() const { return col_idx_.data(); }
    const idx*  row_ptr() const { return row_ptr_.data(); }

private:
    idx n_rows_, n_cols_;
    std::vector<real> vals_;
    std::vector<idx>  col_idx_;
    std::vector<idx>  row_ptr_;   // size n_rows_ + 1
};

/// @brief y = A * x
void sparse_matvec(const SparseMatrix& A, const Vector& x, Vector& y);

} // namespace num
