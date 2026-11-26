/// @file linear/factorization/woodbury.hpp
/// @brief Low-rank corrected solves against a retained factorization.
#pragma once

#include "blas/matrix_ops.hpp"
#include "container/matrix.hpp"
#include "container/matrix_expr.hpp"
#include "linear/factorization/lu.hpp"
#include "linear/matrix_properties.hpp"
#include "linear/matrix_utils.hpp"
#include "linear/solve.hpp"
#include "linear/solvers/solver_result.hpp"
#include "linear/sparse/sparse.hpp"
#include "operator/concepts.hpp"
#include <algorithm>
#include <cmath>
#include <concepts>
#include <limits>
#include <optional>
#include <stdexcept>
#include <utility>

namespace num {

/// @brief Change of a matrix by a low-rank product:
/// \f$A_{\mathrm{new}} = A_{\mathrm{base}} + P Q^T\f$.
struct low_rank_update {
    mat<real> left;  ///< P, of shape n by p.
    mat<real> right; ///< Q, of shape n by p.
};

namespace detail {

struct thin_qr_blocks {
    mat<real> Q;
    mat<real> R;
};

[[nodiscard]] inline thin_qr_blocks thin_modified_gram_schmidt(const mat<real> &A) {
    const idx rows = A.rows(), columns = A.cols();
    mat<real> Q(rows, columns, 0.0), R(columns, columns, 0.0);
    vec<real> column(rows, 0.0);
    idx rank = 0;
    for (idx j = 0; j < columns; ++j) {
        real original_norm = 0.0;
        for (idx i = 0; i < rows; ++i) {
            column[i] = A(i, j);
            original_norm += column[i] * column[i];
        }
        original_norm = std::sqrt(original_norm);
        for (idx pass = 0; pass < 2; ++pass) {
            for (idx q = 0; q < rank; ++q) {
                real coefficient = 0.0;
                for (idx i = 0; i < rows; ++i)
                    coefficient += Q(i, q) * column[i];
                R(q, j) += coefficient;
                for (idx i = 0; i < rows; ++i)
                    column[i] -= coefficient * Q(i, q);
            }
        }
        real norm = 0.0;
        for (real value : column)
            norm += value * value;
        norm = std::sqrt(norm);
        if (norm <= 64.0 * std::numeric_limits<real>::epsilon() * std::max(real(1), original_norm))
            continue;
        R(rank, j) = norm;
        for (idx i = 0; i < rows; ++i)
            Q(i, rank) = column[i] / norm;
        ++rank;
    }
    return {std::move(Q), std::move(R)};
}

} // namespace detail

/// @brief Write the difference of two matrices as a low-rank product:
/// \f$B - A = P Q^T\f$, rank \f$\le 2k\f$ for \f$k\f$ changed rows and columns.
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

/// @brief Write the difference of two matrices as a low-rank product with orthogonalized factors:
/// \f$B - A = P Q^T\f$.
[[nodiscard]] inline low_rank_update orthogonalized_low_rank_difference(const spmat &base,
                                                                        const spmat &current,
                                                                        view<const idx> changed) {
    using namespace ops;
    low_rank_update raw = low_rank_difference(base, current, changed);
    auto left = detail::thin_modified_gram_schmidt(raw.left);
    auto right = detail::thin_modified_gram_schmidt(raw.right);
    mat<real> core = left.R * transpose(right.R);
    return {left.Q * core, std::move(right.Q)};
}

/// @brief Solver for a low-rank change of a factored matrix (Woodbury):
/// \f$(A + P Q^T)^{-1} b\f$.
/// \f$x = y - W G^{-1} P^T y\f$ with \f$G = I + P^T W\f$.
/// @warning References the base factorization; it must outlive the solver.
template <factorization F>
class woodbury_solver {
  public:
    woodbury_solver(const F &base, low_rank_update update)
        : base_(&base), left_(std::move(update.left)), right_(std::move(update.right)) {
        using namespace ops;
        solve_transpose(*base_, right_, transpose_right_);
        mat<real> reduced_transpose = identity(rank()) + transpose(left_) * transpose_right_;
        reduced_transpose_ = lu(reduced_transpose);
        reduced_ = lu(transpose(reduced_transpose));
        if (reduced_transpose_.singular || reduced_.singular) {
            throw std::runtime_error("the Woodbury correction is singular");
        }
    }

    /// @brief Return the rank \f$p\f$ of the change.
    [[nodiscard]] idx rank() const { return left_.cols(); }
    /// @brief Return the order \f$n\f$.
    [[nodiscard]] idx size() const { return left_.rows(); }

    /// @brief Return the left factor \f$P\f$.
    [[nodiscard]] const mat<real> &left() const { return left_; }
    /// @brief Return \f$W = A^{-T} Q\f$.
    [[nodiscard]] const mat<real> &transpose_right() const { return transpose_right_; }

    /// @brief Return \f$A^{-1} P\f$, computed on first use.
    [[nodiscard]] const mat<real> &inverse_left() const {
        if (!inverse_left_) {
            inverse_left_.emplace();
            solve(*base_, left_, *inverse_left_);
        }
        return *inverse_left_;
    }

    /// @brief Return \f$A^{-T} W\f$, computed on first use.
    [[nodiscard]] const mat<real> &transpose_right_squared() const {
        if (!transpose_right_squared_) {
            transpose_right_squared_.emplace();
            solve_transpose(*base_, transpose_right_, *transpose_right_squared_);
        }
        return *transpose_right_squared_;
    }

    /// @brief Solve from the right: \f$Y \leftarrow Y G^{-1}\f$.
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
            num::lu_solve(row, reduced_transpose_.LU.data(), reduced_transpose_.swaps.data(),
                          scratch.data(), p);
        }
    }

    /// @brief Solve with the changed matrix:
    /// \f$x \leftarrow (A + P Q^T)^{-1} b\f$. `x` may alias `b`.
    template <class RHS>
    friend void solve(const woodbury_solver &W, const RHS &b, RHS &x) {
        using namespace ops;
        RHS y;
        solve(*W.base_, b, y);
        RHS coefficients;
        solve(W.reduced_, transpose(W.right_) * y, coefficients);
        x = y - W.inverse_left() * coefficients;
    }

    /// @brief Solve the transposed system with the changed matrix:
    /// \f$x \leftarrow (A + P Q^T)^{-T} b\f$. `x` may alias `b`.
    template <class RHS>
    friend void solve_transpose(const woodbury_solver &W, const RHS &b, RHS &x) {
        using namespace ops;
        RHS y;
        solve_transpose(*W.base_, b, y);
        RHS coefficients;
        solve(W.reduced_transpose_, transpose(W.left_) * y, coefficients);
        x = y - W.transpose_right_ * coefficients;
    }

    /// @brief Update the diagonal of the inverse for the change:
    /// \f$\mathrm{diag}(A_{\mathrm{new}}^{-1})\f$.
    [[nodiscard]] vec<real> inverse_diagonal(view<const real> base_diagonal) const {
        if (base_diagonal.size() != size()) {
            throw std::invalid_argument("the base diagonal has the wrong size");
        }
        mat<real> coefficients;
        solve(reduced_, transpose(transpose_right_), coefficients);
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
    lu_result<real> reduced_, reduced_transpose_;
    mutable std::optional<mat<real>> inverse_left_, transpose_right_squared_;
};

template <factorization F>
woodbury_solver(const F &, low_rank_update) -> woodbury_solver<F>;

/// @brief Reusable buffers for `update_inverse_rows`.
struct inverse_rows_workspace {
    mat<real> first, second;
    vec<real> scratch;
};

/// @brief Update selected rows of \f$A^{-1}\f$ and \f$A^{-2}\f$ for a low-rank change.
template <factorization F>
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
