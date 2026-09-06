/// @file linear/solvers/ilu.hpp
/// @brief Incomplete LU preconditioner with zero fill-in, ILU(0).
///
/// The workhorse preconditioner for nonsymmetric sparse systems, and the reason
/// GMRES is usable on them at all. It factors \f$A \approx LU\f$ subject to the
/// constraint that \f$L\f$ and \f$U\f$ occupy exactly \f$A\f$'s own sparsity
/// pattern — every entry the elimination would create outside that pattern is
/// discarded. So the factors cost the same memory as the matrix, the setup is a
/// single sweep, and applying the preconditioner is two sparse triangular
/// solves.
///
/// ### What it does not promise
///
/// Dropping fill-in is a genuine approximation, and how good an approximation
/// depends on the matrix. ILU(0) is reliable on diagonally dominant and M-matrix
/// systems, which covers most convection-diffusion discretizations. It can
/// break down outright on an indefinite or strongly non-diagonally-dominant
/// matrix — a pivot reaches zero — and the constructor reports that as an error
/// rather than producing a preconditioner that quietly amplifies the residual.
/// It is also *not* symmetric even when \f$A\f$ is, so it carries no
/// self-adjointness claim and must not be handed to PCG or MINRES; those need
/// `approx_chol_preconditioner`, `chebyshev_preconditioner`, or Jacobi.
///
/// All arithmetic lives in `kernel/sparse.hpp` (`ilu0_factor`, `csr_lu_solve`);
/// this class owns the storage, the evidence, and the error reporting.
#pragma once

#include "container/vector.hpp"
#include "core/math/models.hpp"
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
    using domain_type = vec;
    using codomain_type = vec;
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
    void apply(const vec &r, vec &z) const {
        if (r.size() != n_) {
            throw std::invalid_argument("ilu0: dimension mismatch");
        }
        if (z.size() != n_) {
            z = vec(n_, 0.0);
        }
        kernel::csr_lu_solve(z.data(), values_.data(), row_ptr_.data(), col_idx_.data(),
                                  diagonal_.data(), r.data(), n_);
    }

  private:
    idx n_ = 0;
    std::vector<real> values_;
    std::vector<idx> col_idx_;
    std::vector<idx> row_ptr_;
    std::vector<idx> diagonal_;
    std::vector<idx> scratch_;
};

/// @brief Build an ILU(0) preconditioner for a sparse matrix.
[[nodiscard]] inline ilu0_preconditioner make_ilu0_preconditioner(const spmat &A) {
    return ilu0_preconditioner(A);
}

} // namespace num

namespace num::math {

template <>
struct claims_of<ilu0_preconditioner> {
    using type = type_list<law::linear_map>;
};

namespace detail {

template <>
struct domain_of<ilu0_preconditioner, void> {
    using type = vec;
};

template <>
struct codomain_of<ilu0_preconditioner, void> {
    using type = vec;
};

} // namespace detail

} // namespace num::math
