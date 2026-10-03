/// @file linear/solvers/ilu.hpp
/// @brief Incomplete LU preconditioner with zero fill-in, ILU(0).
///
/// L and U keep A's sparsity pattern, so they cost the matrix's memory and applying them is two
/// triangular solves. Reliable on diagonally dominant and M-matrix systems; on others a pivot
/// can vanish, which the constructor reports. It is not symmetric, so it cannot precondition
/// PCG or MINRES.
///
/// The arithmetic is in the raw-pointer `ilu0_factor` and `csr_lu_solve` below, and the class
/// owns the storage and the errors.
#pragma once

#include "container/vector.hpp"
#include "core/math/laws.hpp"
#include "core/types.hpp"
#include "kernel/kernel.hpp"
#include "linear/sparse/sparse.hpp"
#include <stdexcept>
#include <vector>

namespace num {

// Sparse incomplete factorization
//
// ILU(0) keeps A's pattern exactly, so it rewrites the value array in place. L's unit
// diagonal is implicit, and L's strict lower part shares the array with U.

/// @brief Locate each row's diagonal entry in a CSR pattern.
///
/// Requires column indices sorted within each row.
///
/// @param diagonal Output, size n: index into `col_idx` of entry (i,i).
/// @param row_ptr CSR row offsets, size n+1.
/// @param col_idx CSR column indices, sorted within each row.
/// @param n Number of rows.
/// @return False if some row has no diagonal entry, which ILU(0) cannot proceed without.
template <std::integral Index>
[[nodiscard]] NUM_K_AINLINE bool csr_diagonal_positions(Index *NUM_K_RESTRICT diagonal,
                                                        const Index *NUM_K_RESTRICT row_ptr,
                                                        const Index *NUM_K_RESTRICT col_idx,
                                                        std::type_identity_t<Index> n) noexcept {
    for (Index i = 0; i < n; ++i) {
        const Index end = row_ptr[i + 1];
        Index found = end;
        for (Index k = row_ptr[i]; k < end; ++k) {
            if (col_idx[k] == i) {
                found = k;
                break;
            }
        }
        if (found == end) {
            return false;
        }
        diagonal[i] = found;
    }
    return true;
}

/// @brief In-place ILU(0) factorization of a CSR value array.
///
/// `scratch` maps columns to positions in the row being eliminated, so matching a pivot row's
/// column is O(1). It is cleared on exit from each row.
///
/// @param val CSR values, overwritten with the combined factors.
/// @param row_ptr CSR row offsets, size n+1. Not modified.
/// @param col_idx CSR column indices, sorted within each row. Not modified.
/// @param diagonal Diagonal positions from `csr_diagonal_positions`.
/// @param scratch Workspace of size n; contents on entry and exit are irrelevant.
/// @param n Number of rows.
/// @return False if a pivot was zero or non-finite, leaving `val` partially overwritten.
template <std::floating_point T, std::integral Index>
[[nodiscard]] inline bool
ilu0_factor(T *NUM_K_RESTRICT val, const Index *NUM_K_RESTRICT row_ptr,
            const Index *NUM_K_RESTRICT col_idx, const Index *NUM_K_RESTRICT diagonal,
            Index *NUM_K_RESTRICT scratch, std::type_identity_t<Index> n) noexcept {
    constexpr Index unmarked = static_cast<Index>(-1);
    for (Index i = 0; i < n; ++i) {
        scratch[i] = unmarked;
    }

    for (Index i = 0; i < n; ++i) {
        const Index row_begin = row_ptr[i];
        const Index row_end = row_ptr[i + 1];
        for (Index k = row_begin; k < row_end; ++k) {
            scratch[col_idx[k]] = k;
        }

        // Columns strictly left of the diagonal are the L part of this row.
        for (Index k = row_begin; k < diagonal[i]; ++k) {
            const Index j = col_idx[k];
            const T pivot = val[diagonal[j]];
            if (pivot == T(0) || !std::isfinite(pivot)) {
                return false;
            }
            const T multiplier = val[k] / pivot;
            val[k] = multiplier;
            // only columns already in row i: this is what makes it ILU(0)
            for (Index p = diagonal[j] + 1; p < row_ptr[j + 1]; ++p) {
                const Index target = scratch[col_idx[p]];
                if (target != unmarked) {
                    val[target] -= multiplier * val[p];
                }
            }
        }

        const T pivot = val[diagonal[i]];
        if (pivot == T(0) || !std::isfinite(pivot)) {
            return false;
        }
        for (Index k = row_begin; k < row_end; ++k) {
            scratch[col_idx[k]] = unmarked;
        }
    }
    return true;
}

/// @brief Solve \f$LUx = b\f$ for factors packed by `ilu0_factor`.
///
/// Forward substitution against the implicit unit-diagonal L, then backward
/// substitution against U. `x` may alias `b`.
template <std::floating_point T, std::integral Index>
NUM_K_AINLINE void
csr_lu_solve(T *NUM_K_RESTRICT x, const T *NUM_K_RESTRICT val, const Index *NUM_K_RESTRICT row_ptr,
             const Index *NUM_K_RESTRICT col_idx, const Index *NUM_K_RESTRICT diagonal, const T *b,
             std::type_identity_t<Index> n) noexcept {
    for (Index i = 0; i < n; ++i) {
        T sum = b[i];
        for (Index k = row_ptr[i]; k < diagonal[i]; ++k) {
            sum -= val[k] * x[col_idx[k]];
        }
        x[i] = sum; // L has a unit diagonal, so no division here
    }
    for (Index i = n; i-- > 0;) {
        T sum = x[i];
        for (Index k = diagonal[i] + 1; k < row_ptr[i + 1]; ++k) {
            sum -= val[k] * x[col_idx[k]];
        }
        x[i] = sum / val[diagonal[i]];
    }
}

} // namespace num

namespace num {

/// @brief ILU(0) preconditioner: \f$M^{-1} r\f$ by two sparse triangular solves.
///
/// Holds its own copy of the factored values, so the source matrix need not
/// outlive it. Allocates nothing after construction.
class ilu0_preconditioner final {
  public:
    using domain_type = vec<real>;
    using codomain_type = vec<real>;
    // Deliberately no property claims. An incomplete LU is not self-adjoint even
    // for a symmetric A, so PCG and MINRES will not accept it -- which is the
    // correct outcome, enforced by the type system rather than by documentation.

    /// @brief Factor `A` in place over its own pattern.
    /// @throws std::invalid_argument If `A` is not square, or a row has no diagonal entry.
    /// @throws std::runtime_error If the factorization reaches a zero or non-finite pivot.
    explicit ilu0_preconditioner(const spmat &A)
        : n_(A.n_rows()), values_(A.values(), A.values() + A.nnz()),
          col_idx_(A.col_idx(), A.col_idx() + A.nnz()),
          row_ptr_(A.row_ptr(), A.row_ptr() + A.n_rows() + 1), diagonal_(A.n_rows(), 0),
          scratch_(A.n_rows(), 0) {
        if (A.n_rows() != A.n_cols()) {
            throw std::invalid_argument("ilu0: matrix must be square");
        }
        if (n_ == 0) {
            return;
        }
        if (!num::csr_diagonal_positions(diagonal_.data(), row_ptr_.data(), col_idx_.data(), n_)) {
            throw std::invalid_argument(
                "ilu0: every row must have a stored diagonal entry, including explicit zeros");
        }
        if (!num::ilu0_factor(values_.data(), row_ptr_.data(), col_idx_.data(), diagonal_.data(),
                              scratch_.data(), n_)) {
            throw std::runtime_error(
                "ilu0: zero or non-finite pivot; the matrix is too far from diagonally dominant "
                "for zero fill-in");
        }
    }

    [[nodiscard]] idx rows() const noexcept { return n_; }
    [[nodiscard]] idx cols() const noexcept { return n_; }
    [[nodiscard]] idx nnz() const noexcept { return values_.size(); }

    /// @brief Apply \f$z \leftarrow (LU)^{-1} r\f$.
    void apply(const vec<real> &r, vec<real> &z) const {
        if (r.size() != n_) {
            throw std::invalid_argument("ilu0: dimension mismatch");
        }
        if (z.size() != n_) {
            z = vec<real>(n_, 0.0);
        }
        num::csr_lu_solve(z.data(), values_.data(), row_ptr_.data(), col_idx_.data(),
                          diagonal_.data(), r.data(), n_);
    }

  private:
    idx n_ = 0;
    array<real> values_;
    array<idx> col_idx_;
    array<idx> row_ptr_;
    array<idx> diagonal_;
    array<idx> scratch_;
};

/// @brief Build an ILU(0) preconditioner for a sparse matrix.
[[nodiscard]] inline ilu0_preconditioner make_ilu0_preconditioner(const spmat &A) {
    return ilu0_preconditioner(A);
}

} // namespace num
