/// @file linear/factorization/reuse.hpp
/// @brief Refactorization after a few rows and columns change, reusing the previous factor.
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

// Returned by `first_changed_block` when no leading block can be kept.
inline constexpr idx no_reusable_block = static_cast<idx>(-1);

// The first block of the new ordering that `changed` touches, provided every block before
// it has the same rows in the same order as in `Z`; else `no_reusable_block`.
template <typename F>
[[nodiscard]] inline idx first_changed_block(const F &Z, const block_layout &layout,
                                             view<const idx> changed) {
    array<idx> old_position(Z.size);
    for (idx position = 0; position < Z.size; ++position) {
        old_position[Z.order[position]] = position;
    }

    idx first = layout.offsets.size() - 1;
    for (idx row : changed) {
        first = std::min(first, layout.block_of[row]);
        const auto boundary =
            std::upper_bound(Z.offsets.begin(), Z.offsets.end(), old_position[row]);
        first = std::min(first, static_cast<idx>(boundary - Z.offsets.begin() - 1));
    }
    if (first > Z.blocks() || first + 1 > layout.offsets.size()) {
        return no_reusable_block;
    }
    for (idx k = 0; k <= first; ++k) {
        if (Z.offsets[k] != layout.offsets[k]) {
            return no_reusable_block;
        }
    }
    for (idx position = 0; position < layout.offsets[first]; ++position) {
        if (Z.order[position] != layout.order[position]) {
            return no_reusable_block;
        }
    }
    return first;
}

// Skeel's condition number || |A^{-1}| |A| ||_inf, or infinity for a singular A.
// Unlike the plain condition number it ignores the scaling of the rows of A.
inline real skeel_condition(const mat<real> &A) {
    const lu_result<real> factor = lu(A);
    if (factor.singular) {
        return std::numeric_limits<real>::infinity();
    }
    mat<real> inverse;
    solve(factor, identity(A.rows()), inverse);
    real worst = 0.0;
    for (idx i = 0; i < A.rows(); ++i) {
        real row = 0.0;
        for (idx j = 0; j < A.cols(); ++j) {
            for (idx k = 0; k < A.rows(); ++k) {
                row += std::abs(inverse(i, k)) * std::abs(A(k, j));
            }
        }
        worst = num::max(worst, row);
    }
    return worst;
}

} // namespace detail

/// @brief Woodbury-corrected factor: the last full factorization plus a low-rank correction.
template <factorization Base>
class corrected_factor {
  public:
    static constexpr idx max_changed = 64;
    static constexpr real tolerance = 1e-8;

    /// @brief Whether the factor was reused rather than recomputed.
    [[nodiscard]] bool reused() const { return reused_; }
    [[nodiscard]] idx size() const { return base_R_->n_rows(); }

    template <class RHS>
    friend void solve(const corrected_factor &Z, const RHS &rhs, RHS &out) {
        if (Z.correction_) {
            solve(*Z.correction_, rhs, out);
        } else {
            solve(*Z.base_, rhs, out);
        }
    }

    template <class RHS>
    friend void solve_transpose(const corrected_factor &Z, const RHS &rhs, RHS &out) {
        if (Z.correction_) {
            solve_transpose(*Z.correction_, rhs, out);
        } else {
            solve_transpose(*Z.base_, rhs, out);
        }
    }

    friend vec<real> inverse_diagonal(const corrected_factor &Z, idx n) {
        if (n != Z.size())
            throw std::invalid_argument("inverse diagonal size does not match the factor");
        if (!Z.base_diagonal_)
            Z.base_diagonal_ = std::make_shared<const vec<real>>(inverse_diagonal(*Z.base_, n));
        if (Z.correction_)
            return Z.correction_->inverse_diagonal(*Z.base_diagonal_);
        return *Z.base_diagonal_;
    }

    template <class Fresh>
    static corrected_factor update(const spmat &R, const corrected_factor *previous,
                                   view<const idx> changed, Fresh &&fresh) {
        const idx n = R.n_rows();
        corrected_factor next;
        if (previous && previous->size() == n) {
            next.base_ = previous->base_;
            next.base_R_ = previous->base_R_;
            next.base_diagonal_ = previous->base_diagonal_;
            array<bool> differs(n, false);
            for (idx slot : previous->differing_) {
                differs[slot] = true;
            }
            next.differing_ = previous->differing_;
            for (idx slot : changed) {
                if (slot >= n) {
                    throw std::out_of_range("factor reuse: changed slot is outside R");
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
            if (next.differing_.size() <= max_changed && next.correct(R)) {
                next.reused_ = true;
                return next;
            }
        }
        next.base_ = std::make_shared<const Base>(fresh());
        next.base_R_ = std::make_shared<const spmat>(R);
        next.differing_.clear();
        next.correction_.reset();
        next.base_diagonal_.reset();
        next.reused_ = false;
        return next;
    }

  private:
    // A Woodbury correction for R, kept only if it is accurate. With
    // R^T = R_b^T + Q P^T and G = I + P^T W, the corrected solve of R^T x = b
    // has residual Q (P^T y - G c), so its accuracy is that of the p-by-p solve
    // with G.
    bool correct(const spmat &R) {
        try {
            correction_.emplace(*base_,
                                orthogonalized_low_rank_difference(*base_R_, R, differing_));
        } catch (const std::runtime_error &) {
            correction_.reset();
            return false;
        }
        using namespace ops;
        const mat<real> &P = correction_->left();
        const mat<real> &W = correction_->transpose_right();
        const mat<real> G = identity(P.cols()) + transpose(P) * W;
        if (!(detail::skeel_condition(G) * std::numeric_limits<real>::epsilon() <= tolerance)) {
            correction_.reset();
            return false;
        }
        return true;
    }

    std::shared_ptr<const Base> base_;
    std::shared_ptr<const spmat> base_R_;
    mutable std::shared_ptr<const vec<real>> base_diagonal_;
    array<idx> differing_;
    std::optional<woodbury_solver<Base>> correction_;
    bool reused_ = false;
};

using corrected_lu = corrected_factor<lu_result<real>>;
using corrected_cholesky = corrected_factor<similar_factor<cholesky_result>>;

/// @brief Unpivoted LU, Woodbury-corrected from `previous` when few slots change: \f$A = L U\f$.
inline corrected_lu lu(const spmat &R, no_pivot_structure, const corrected_lu *previous,
                       view<const idx> changed) {
    return corrected_lu::update(R, previous, changed, [&] {
        lu_result<real> base = lu(R, no_pivot);
        if (base.singular) {
            throw std::runtime_error("lu: no-pivot factorization is singular");
        }
        return base;
    });
}

/// @brief Cholesky of a diagonal similarity, Woodbury-corrected: \f$H A H^{-1} = L L^T\f$.
inline corrected_cholesky cholesky(const spmat &R, view<const real> h,
                                   const corrected_cholesky *previous, view<const idx> changed) {
    return corrected_cholesky::update(R, previous, changed, [&] { return cholesky(R, h); });
}

/// @brief Block LU with the number of reused leading blocks.
struct suffix_block_lu {
    block_lu_factor factor;
    idx reused_blocks = 0;

    [[nodiscard]] bool reused() const { return reused_blocks > 0; }
    [[nodiscard]] idx size() const { return factor.size; }
};

/// @brief Block LU reusing the unchanged prefix of `previous`.
inline suffix_block_lu lu(const spmat &R, block_structure structure,
                          const suffix_block_lu *previous, view<const idx> changed) {
    const idx n = R.n_rows();
    if (previous && previous->size() == n && structure.levels.size() == n) {
        const detail::block_layout layout = detail::build_block_order(structure.levels);
        const idx first = detail::first_changed_block(previous->factor, layout, changed);
        if (first != detail::no_reusable_block) {
            try {
                return {refactor_block_lu_suffix(R, structure.levels, previous->factor, first),
                        first};
            } catch (const std::exception &) {
                // The kept prefix no longer factors cleanly; fall through to a fresh factor.
            }
        }
    }
    return {lu(R, structure), 0};
}

template <class RHS>
inline void solve(const suffix_block_lu &Z, const RHS &rhs, RHS &out) {
    solve(Z.factor, rhs, out);
}

template <class RHS>
inline void solve_transpose(const suffix_block_lu &Z, const RHS &rhs, RHS &out) {
    solve_transpose(Z.factor, rhs, out);
}

/// @brief Block Cholesky with the number of reused leading blocks.
struct suffix_block_cholesky {
    similar_factor<block_cholesky_factor> factor;
    idx reused_blocks = 0;

    [[nodiscard]] bool reused() const { return reused_blocks > 0; }
    [[nodiscard]] idx size() const { return factor.factor.size; }
};

/// @brief Block Cholesky of \f$H A H^{-1}\f$, reusing the unchanged prefix of `previous`.
inline suffix_block_cholesky cholesky(const spmat &R, block_structure structure, view<const real> h,
                                      const suffix_block_cholesky *previous,
                                      view<const idx> changed) {
    vec<real> weights = detail::checked_weights(h);
    const spmat symmetric = sparse_diagonal_similarity(R, detail::reciprocal(weights));
    const idx n = R.n_rows();
    if (previous && previous->size() == n && structure.levels.size() == n) {
        const detail::block_layout layout = detail::build_block_order(structure.levels);
        const idx first = detail::first_changed_block(previous->factor.factor, layout, changed);
        bool same_weights = first != detail::no_reusable_block;
        if (same_weights) {
            for (idx position = 0; position < layout.offsets[first]; ++position) {
                const idx row = layout.order[position];
                same_weights = same_weights && previous->factor.h[row] == weights[row];
            }
        }
        if (same_weights) {
            try {
                similar_factor<block_cholesky_factor> factor{
                    refactor_block_cholesky_suffix(symmetric, structure.levels,
                                                   previous->factor.factor, first),
                    std::move(weights)};
                return {std::move(factor), first};
            } catch (const std::exception &) {
                // The retained prefix is unusable; factor the complete matrix below.
            }
        }
    }
    return {cholesky(R, structure, weights), 0};
}

template <class RHS>
inline void solve(const suffix_block_cholesky &Z, const RHS &rhs, RHS &out) {
    solve(Z.factor, rhs, out);
}

template <class RHS>
inline void solve_transpose(const suffix_block_cholesky &Z, const RHS &rhs, RHS &out) {
    solve_transpose(Z.factor, rhs, out);
}

} // namespace num
