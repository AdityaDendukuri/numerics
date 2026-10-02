/// @file linear/solvers/ilu.hpp
/// @brief Incomplete LU preconditioner with zero fill-in, ILU(0).
///
/// L and U keep A's sparsity pattern, so they cost the matrix's memory and applying them is two
/// triangular solves. Reliable on diagonally dominant and M-matrix systems; on others a pivot
/// can vanish, which the constructor reports. It is not symmetric, so it cannot precondition
/// PCG or MINRES.
///
/// The arithmetic is in `kernel/sparse.hpp`; this class owns the storage and the errors.
#pragma once

#include "container/vector.hpp"
#include "core/math/laws.hpp"
#include "core/types.hpp"
#include "kernel/kernel.hpp"
#include "linear/sparse/sparse.hpp"
#include <stdexcept>
#include <vector>

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
        if (!kernel::csr_diagonal_positions(diagonal_.data(), row_ptr_.data(),
                                                 col_idx_.data(), n_)) {
            throw std::invalid_argument(
                "ilu0: every row must have a stored diagonal entry, including explicit zeros");
        }
        if (!kernel::ilu0_factor(values_.data(), row_ptr_.data(), col_idx_.data(),
                                      diagonal_.data(), scratch_.data(), n_)) {
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
        kernel::csr_lu_solve(z.data(), values_.data(), row_ptr_.data(), col_idx_.data(),
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

