/// @file linear/solvers/ilu.hpp
/// @brief Incomplete LU preconditioner, zero fill: ILU(0).
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

/// @brief Find the position of each row's diagonal entry; returns false if a row has none.
/// Column indices must be sorted.
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

/// @brief Factor by incomplete LU on the pattern of \f$A\f$, in place (ILU(0)):
/// \f$A \approx L U\f$.
/// Returns false on a zero or non-finite pivot.
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

/// @brief Solve with sparse LU factors: \f$x \leftarrow U^{-1} L^{-1} b\f$. `x` may alias `b`.
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

/// @brief Preconditioner by incomplete LU factorization (ILU(0)): \f$M^{-1} r = (L U)^{-1} r\f$.
/// @note Reliable on diagonally dominant and M-matrix systems; elsewhere a pivot can vanish.
/// @warning Not symmetric, so it cannot precondition PCG or MINRES; use it with GMRES.
class ilu0_preconditioner final {
  public:
    using domain_type = vec<real>;
    using codomain_type = vec<real>;
    // Deliberately no property claims. An incomplete LU is not self-adjoint even
    // for a symmetric A, so PCG and MINRES will not accept it -- which is the
    // correct outcome, enforced by the type system rather than by documentation.

    /// @brief Factor `A` by incomplete LU.
    /// @throws std::invalid_argument If `A` is not square or a row lacks a diagonal entry.
    /// @throws std::runtime_error On a zero or non-finite pivot.
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

    /// @brief Apply the preconditioner: \f$z \leftarrow (L U)^{-1} r\f$.
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

/// @brief Construct an incomplete LU preconditioner of `A`.
[[nodiscard]] inline ilu0_preconditioner make_ilu0_preconditioner(const spmat &A) {
    return ilu0_preconditioner(A);
}

} // namespace num
