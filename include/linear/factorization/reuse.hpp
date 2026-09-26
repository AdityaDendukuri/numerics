/// @file linear/factorization/reuse.hpp
/// @brief Factor R again from the factor of a matrix that differs at a few slots.
///
/// Each overload takes the factor of the previous matrix, or null, and the
/// slots whose row and column changed; the two matrices agree everywhere else.
///
/// - `no_pivot`: Woodbury corrections against the last complete factorization
///   while at most `corrected_lu::max_changed` slots differ from it and the
///   correction's condition number times machine epsilon stays below
///   `corrected_lu::tolerance`.
/// - `blocks(levels)`: the unchanged block prefix is copied and the suffix
///   refactored.
///
/// Every other case factors R from scratch; `reused()` says which happened.
#pragma once

#include "linear/factorization/factor.hpp"
#include "linear/factorization/woodbury.hpp"
#include "linear/matrix_utils.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>

namespace num {

namespace detail {

// Skeel's condition number || |A^{-1}| |A| ||_inf, or infinity for a singular A.
// Unlike the plain condition number it ignores the scaling of the rows of A.
inline real skeel_condition(const mat &A) {
    const lu_result factor = lu(assume_square(A));
    if (factor.singular) {
        return std::numeric_limits<real>::infinity();
    }
    mat inverse;
    lu_solve(factor, identity(A.rows()), inverse);
    real worst = 0.0;
    for (idx i = 0; i < A.rows(); ++i) {
        real row = 0.0;
        for (idx j = 0; j < A.cols(); ++j) {
            for (idx k = 0; k < A.rows(); ++k) {
                row += std::abs(inverse(i, k)) * std::abs(A(k, j));
            }
        }
        worst = std::max(worst, row);
    }
    return worst;
}

} // namespace detail

/// A no-pivot LU of R, held as the last complete factorization and a Woodbury
/// correction for the slots that changed since.
class corrected_lu {
  public:
    static constexpr idx max_changed = 3;
    static constexpr real tolerance = 1e-8;

    /// True when R was not factored from scratch.
    [[nodiscard]] bool reused() const { return reused_; }
    [[nodiscard]] idx size() const { return base_->size(); }

    template <class RHS>
    void solve(const RHS &rhs, RHS &out) const {
        if (correction_) {
            out = correction_->solve(rhs);
        } else {
            num::solve(*base_, rhs, out);
        }
    }

    template <class RHS>
    void solve_transpose(const RHS &rhs, RHS &out) const {
        if (correction_) {
            out = correction_->solve_transpose(rhs);
        } else {
            num::solve_transpose(*base_, rhs, out);
        }
    }

  private:
    friend corrected_lu lu(const spmat &R, no_pivot_structure, const corrected_lu *previous,
                           view<const idx> changed);

    // A Woodbury correction for R, kept only if it is accurate. With
    // R^T = R_b^T + Q P^T and G = I + P^T W, the corrected solve of R^T x = b
    // has residual Q (P^T y - G c), so its accuracy is that of the p-by-p solve
    // with G. The base is a no-pivot LU of an M-matrix, which is stable.
    bool correct(const spmat &R) {
        try {
            correction_.emplace(*base_, low_rank_difference(*base_R_, R, differing_));
        } catch (const std::runtime_error &) {
            correction_.reset();
            return false;
        }
        using namespace ops;
        const mat &P = correction_->left();
        const mat &W = correction_->transpose_right();
        const mat G = identity(P.cols()) + transpose(P) * W;
        if (!(detail::skeel_condition(G) * std::numeric_limits<real>::epsilon() <= tolerance)) {
            correction_.reset();
            return false;
        }
        return true;
    }

    std::shared_ptr<const no_pivot_lu> base_;
    std::shared_ptr<const spmat> base_R_;
    array<idx> differing_;
    std::optional<woodbury_solver<no_pivot_lu>> correction_;
    bool reused_ = false;
};

/// @brief No-pivot LU of R, corrected from `previous` when few slots differ.
inline corrected_lu lu(const spmat &R, no_pivot_structure, const corrected_lu *previous,
                       view<const idx> changed) {
    const idx n = R.n_rows();
    corrected_lu next;
    if (previous && previous->size() == n) {
        next.base_ = previous->base_;
        next.base_R_ = previous->base_R_;
        array<bool> differs(n, false);
        for (idx slot : previous->differing_) {
            differs[slot] = true;
        }
        next.differing_ = previous->differing_;
        for (idx slot : changed) {
            if (slot >= n) {
                throw std::out_of_range("lu: changed slot is outside R");
            }
            if (!differs[slot]) {
                differs[slot] = true;
                next.differing_.push_back(slot);
            }
        }
        if (next.differing_.empty()) {
            next.reused_ = true;
            return next;
        }
        if (next.differing_.size() <= corrected_lu::max_changed && next.correct(R)) {
            next.reused_ = true;
            return next;
        }
    }
    no_pivot_lu base = lu(R, no_pivot);
    if (base.singular) {
        throw std::runtime_error("lu: no-pivot factorization is singular");
    }
    next.base_ = std::make_shared<const no_pivot_lu>(std::move(base));
    next.base_R_ = std::make_shared<const spmat>(R);
    next.differing_.clear();
    next.correction_.reset();
    next.reused_ = false;
    return next;
}

/// A block LU of R and the number of leading blocks copied from the previous factor.
struct suffix_block_lu {
    block_lu_factor factor;
    idx reused_blocks = 0;

    [[nodiscard]] bool reused() const { return reused_blocks > 0; }
    [[nodiscard]] idx size() const { return factor.size; }
};

/// @brief Block LU of R, copying the unchanged prefix of `previous`.
inline suffix_block_lu lu(const spmat &R, block_structure structure,
                          const suffix_block_lu *previous, view<const idx> changed) {
    if (previous && previous->size() == R.n_rows()) {
        suffix_reuse_report report;
        if (auto updated = refactor_suffix(previous->factor, R, structure, changed, &report)) {
            return {std::move(*updated), report.reused_blocks};
        }
    }
    return {lu(R, structure), 0};
}

template <class RHS>
inline void solve(const corrected_lu &Z, const RHS &rhs, RHS &out) {
    Z.solve(rhs, out);
}

template <class RHS>
inline void solve_transpose(const corrected_lu &Z, const RHS &rhs, RHS &out) {
    Z.solve_transpose(rhs, out);
}

template <class RHS>
inline void solve(const suffix_block_lu &Z, const RHS &rhs, RHS &out) {
    solve(Z.factor, rhs, out);
}

template <class RHS>
inline void solve_transpose(const suffix_block_lu &Z, const RHS &rhs, RHS &out) {
    solve_transpose(Z.factor, rhs, out);
}

} // namespace num
