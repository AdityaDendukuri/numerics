/// @file linear/factorization/woodbury.hpp
/// @brief Low-rank corrected solves against a retained factorization.
#pragma once

#include "blas/matrix_ops.hpp"
#include "container/matrix.hpp"
#include "container/matrix_expr.hpp"
#include "kernel/factor.hpp"
#include "linear/matrix_properties.hpp"
#include "linear/solvers/solver_result.hpp"
#include "operator/concepts.hpp"
#include "linear/factorization/lu.hpp"
#include "linear/matrix_utils.hpp"
#include "linear/sparse/sparse.hpp"
#include <concepts>
#include <optional>
#include <stdexcept>
#include <utility>

namespace num {

/// An additive change \f$A_{new} = A_{base} + PQ^{T}\f$ of rank at most p.
struct low_rank_update {
    mat<real> left;  ///< P, of shape n by p.
    mat<real> right; ///< Q, of shape n by p.
};

/// @brief A reusable factorization that applies \f$A^{-1}\f$ and \f$A^{-T}\f$.
///
/// Woodbury needs the transpose, which `direct_factorization` does not promise,
/// and it needs out-of-place solves so a corrected result can be formed without
/// destroying the right-hand side. The out-parameter forms may alias their input.
template <class F, class Vec = vec<real>, class Mat = mat<real>>
concept retained_factorization =
    vector_space<Vec> &&
    (requires(const F &factor, const Vec &v, const Mat &m, Vec &vout, Mat &mout) {
        solve(factor, v, vout);
        solve(factor, m, mout);
        solve_transpose(factor, v, vout);
        solve_transpose(factor, m, mout);
    } || requires(const F &factor, const Vec &v, const Mat &m, Vec &vout, Mat &mout) {
        factor.solve(v, vout);
        factor.solve(m, mout);
        factor.solve_transpose(v, vout);
        factor.solve_transpose(m, mout);
    });

namespace detail {

template <class F, class RHS>
void apply_solve(const F &factor, const RHS &rhs, RHS &out) {
    if constexpr (requires { solve(factor, rhs, out); }) {
        solve(factor, rhs, out);
    } else {
        factor.solve(rhs, out);
    }
}

template <class F, class RHS>
void apply_solve_transpose(const F &factor, const RHS &rhs, RHS &out) {
    if constexpr (requires { solve_transpose(factor, rhs, out); }) {
        solve_transpose(factor, rhs, out);
    } else {
        factor.solve_transpose(rhs, out);
    }
}

} // namespace detail

/// @brief Express \f$B - A\f$ as \f$PQ^{T}\f$ when the two differ only in the
/// listed rows and columns.
///
/// Changing k indices of a square matrix touches at most k rows and k columns,
/// so the difference has rank at most 2k. The first k columns of P are the
/// corresponding unit vectors and carry the changed rows in Q; the last k
/// columns of Q are the unit vectors and carry the changed columns in P,
/// excluding entries a changed row already accounts for.
[[nodiscard]] inline low_rank_update low_rank_difference(const spmat &base, const spmat &current,
                                                         view<const idx> changed) {
    const idx n = base.n_rows();
    if (base.n_cols() != n || current.n_rows() != n || current.n_cols() != n) {
        throw std::invalid_argument("a low-rank difference needs equal square matrices");
    }
    const idx count = changed.size();
    array<idx> column_of(n, n);
    for (idx k = 0; k < count; ++k) {
        if (changed[k] >= n) {
            throw std::out_of_range("changed index is outside the matrix");
        }
        column_of[changed[k]] = k;
    }
    mat<real> left(n, 2 * count, 0.0);
    mat<real> right(n, 2 * count, 0.0);
    for (idx k = 0; k < count; ++k) {
        left(changed[k], k) = 1.0;
        right(changed[k], count + k) = 1.0;
    }
    const auto accumulate = [&](const spmat &matrix, real sign) {
        for (idx i = 0; i < n; ++i) {
            for (auto entry = matrix.row_ptr()[i]; entry < matrix.row_ptr()[i + 1]; ++entry) {
                const idx j = static_cast<idx>(matrix.col_idx()[entry]);
                const real value = sign * matrix.values()[entry];
                if (column_of[i] != n) {
                    right(j, column_of[i]) += value;
                } else if (column_of[j] != n) {
                    left(i, count + column_of[j]) += value;
                }
            }
        }
    };
    accumulate(current, 1.0);
    accumulate(base, -1.0);
    return {std::move(left), std::move(right)};
}

/// @brief Solves with \f$A_{base} + PQ^{T}\f$ through the retained factor of
/// \f$A_{base}\f$.
///
/// The Woodbury identity gives \f$x = y - WG^{-1}P^{T}y\f$ with
/// \f$y = A_{base}^{-T}b\f$, \f$W = A_{base}^{-T}Q\f$ and \f$G = I + P^{T}W\f$,
/// and the mirror identity for forward solves. Construction costs one rank-p
/// transpose solve. \f$A_{base}^{-1}P\f$ and \f$A_{base}^{-T}W\f$ are formed on
/// first use, so a caller that only solves transposes never pays for them.
///
/// The base factorization is referenced, not copied, and must outlive this object.
template <retained_factorization F>
class woodbury_solver {
  public:
    woodbury_solver(const F &base, low_rank_update update)
        : base_(&base), left_(std::move(update.left)), right_(std::move(update.right)) {
        using namespace ops;
        detail::apply_solve_transpose(*base_, right_, transpose_right_);
        mat<real> reduced_transpose = identity(rank()) + transpose(left_) * transpose_right_;
        reduced_transpose_ = lu(reduced_transpose);
        reduced_ = lu(transpose(reduced_transpose));
        if (reduced_transpose_.singular || reduced_.singular) {
            throw std::runtime_error("the Woodbury correction is singular");
        }
    }

    /// The rank p of the update.
    [[nodiscard]] idx rank() const { return left_.cols(); }
    /// The order n of the corrected matrix.
    [[nodiscard]] idx size() const { return left_.rows(); }

    /// P.
    [[nodiscard]] const mat<real> &left() const { return left_; }
    /// \f$W = A_{base}^{-T}Q\f$.
    [[nodiscard]] const mat<real> &transpose_right() const { return transpose_right_; }

    /// \f$A_{base}^{-1}P\f$, formed on first use.
    [[nodiscard]] const mat<real> &inverse_left() const {
        if (!inverse_left_) {
            inverse_left_.emplace();
            detail::apply_solve(*base_, left_, *inverse_left_);
        }
        return *inverse_left_;
    }

    /// \f$A_{base}^{-T}W\f$, formed on first use.
    [[nodiscard]] const mat<real> &transpose_right_squared() const {
        if (!transpose_right_squared_) {
            transpose_right_squared_.emplace();
            detail::apply_solve_transpose(*base_, transpose_right_, *transpose_right_squared_);
        }
        return *transpose_right_squared_;
    }

    /// @brief Replace y by \f$yG^{-1}\f$, solving \f$G^{T}y_i^{T} = y_i^{T}\f$ per row.
    void right_solve(mat<real> &y, vec<real> &scratch) const {
        const idx p = rank();
        if (scratch.size() != p) {
            scratch = vec<real>(p, 0.0);
        }
        for (idx i = 0; i < y.rows(); ++i) {
            real *row = y.data() + (i * p);
            for (idx j = 0; j < p; ++j) {
                scratch[j] = row[j];
            }
            kernel::lu_solve(row, reduced_transpose_.LU.data(), reduced_transpose_.piv.data(),
                             scratch.data(), p);
        }
    }

    /// Solve \f$(A_{base} + PQ^{T})^{T}x = b\f$.
    template <class RightHandSide>
    [[nodiscard]] RightHandSide solve_transpose(const RightHandSide &rhs) const {
        using namespace ops;
        RightHandSide y;
        detail::apply_solve_transpose(*base_, rhs, y);
        RightHandSide coefficients;
        lu_solve(reduced_transpose_, transpose(left_) * y, coefficients);
        return y - transpose_right_ * coefficients;
    }

    /// Solve \f$(A_{base} + PQ^{T})x = b\f$.
    template <class RightHandSide>
    [[nodiscard]] RightHandSide solve(const RightHandSide &rhs) const {
        using namespace ops;
        RightHandSide y;
        detail::apply_solve(*base_, rhs, y);
        RightHandSide coefficients;
        lu_solve(reduced_, transpose(right_) * y, coefficients);
        return y - inverse_left() * coefficients;
    }

    /// @brief \f$diag(A_{new}^{-1})\f$ from the base diagonal.
    ///
    /// \f$diag(A_{new}^{-1}) = diag(A_{base}^{-1})
    /// - diag(A_{base}^{-1}PG^{-1}Q^{T}A_{base}^{-1})\f$, which costs no solve
    /// beyond the two rank-p blocks already held.
    [[nodiscard]] vec<real> inverse_diagonal(view<const real> base_diagonal) const {
        if (base_diagonal.size() != size()) {
            throw std::invalid_argument("the base diagonal has the wrong size");
        }
        mat<real> coefficients;
        lu_solve(reduced_, transpose(transpose_right_), coefficients);
        const mat<real> &left_columns = inverse_left();
        vec<real> diagonal(size(), 0.0);
        for (idx i = 0; i < size(); ++i) {
            real correction = 0.0;
            for (idx column = 0; column < rank(); ++column) {
                correction += left_columns(i, column) * coefficients(column, i);
            }
            diagonal[i] = base_diagonal[i] - correction;
        }
        return diagonal;
    }

  private:
    const F *base_;
    mat<real> left_, right_, transpose_right_;
    lu_result reduced_, reduced_transpose_;
    mutable std::optional<mat<real>> inverse_left_, transpose_right_squared_;
};

template <retained_factorization F>
woodbury_solver(const F &, low_rank_update) -> woodbury_solver<F>;

/// Reusable blocks for `update_inverse_rows`, resized on demand.
struct inverse_rows_workspace {
    mat<real> first, second;
    vec<real> scratch;
};

/// @brief Carry selected rows of \f$A^{-1}\f$ and \f$A^{-2}\f$ across a low-rank change.
///
/// With \f$U_b = EA_{base}^{-1}\f$ and \f$V_b = EA_{base}^{-2}\f$ on entry, this
/// leaves the corresponding rows of the corrected inverse and its square in
/// `first` and `second`. Both updates are rank-p, so the cost does not grow with
/// the number of rows carried.
template <retained_factorization F>
void update_inverse_rows(const woodbury_solver<F> &correction, mat<real> &first, mat<real> &second,
                         inverse_rows_workspace &work) {
    const idx rows = first.rows(), p = correction.rank();
    if (work.first.rows() != rows || work.first.cols() != p) {
        work.first = mat<real>(rows, p, 0.0);
        work.second = mat<real>(rows, p, 0.0);
    }
    blas::gemm(1.0, first, false, correction.left(), false, 0.0, work.first);
    correction.right_solve(work.first, work.scratch);
    blas::gemm(-1.0, work.first, false, correction.transpose_right(), true, 1.0, first);
    blas::gemm(-1.0, work.first, false, correction.transpose_right_squared(), true, 1.0, second);
    blas::gemm(1.0, second, false, correction.left(), false, 0.0, work.second);
    correction.right_solve(work.second, work.scratch);
    blas::gemm(-1.0, work.second, false, correction.transpose_right(), true, 1.0, second);
}

} // namespace num
