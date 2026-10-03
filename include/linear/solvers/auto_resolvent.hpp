/// @file linear/solvers/auto_resolvent.hpp
/// @brief Automatic dense/sparse shifted-resolvent selection.
#pragma once

#include "core/types.hpp"
#include "linear/solve.hpp"
#include "linear/solvers/hessenberg_resolvent.hpp"
#include "linear/solvers/sparse_resolvent.hpp"
#include "linear/sparse/sparse.hpp"
#include <optional>
#include <stdexcept>
#include <variant>

namespace num {

struct auto_shifted_lu;

/// Dense/sparse cutoff and optional sparse symbolic symmetry hint.
struct auto_resolvent_options {
    idx dense_limit = 128;
    bool symmetric_pattern = false;
};

/// @brief \f$A\f$ prepared for shifted solves \f$(sI - A)x = b\f$, by Hessenberg reduction at
/// or below `dense_limit` and by sparse analysis above it.
class auto_resolvent {
  public:
    explicit auto_resolvent(const spmat &A, auto_resolvent_options options = {}) {
        if (A.n_rows() <= options.dense_limit) {
            dense_.emplace(A);
            return;
        }
        if (!sparse_resolvent_available()) {
            throw std::runtime_error("large shifted systems require the sparse complex backend");
        }
        sparse_.emplace(A, sparse_resolvent_options{.symmetric_pattern = options.symmetric_pattern});
    }

    [[nodiscard]] idx size() const noexcept { return dense_ ? dense_->size() : sparse_->size(); }

  private:
    std::optional<hessenberg_resolvent> dense_;
    std::optional<sparse_resolvent> sparse_;
    friend auto_shifted_lu shift(const auto_resolvent &R, cplx s);
};

/// @brief \f$sI - A\f$ for one shift, factored by whichever method `auto_resolvent` chose.
struct auto_shifted_lu {
    std::variant<hessenberg_shifted_lu, sparse_shifted_lu> factor;

    [[nodiscard]] idx size() const noexcept {
        return std::visit([](const auto &F) { return F.size(); }, factor);
    }
};

/// @brief Factor \f$sI - A\f$.
[[nodiscard]] inline auto_shifted_lu shift(const auto_resolvent &R, cplx s) {
    if (R.dense_) {
        return {shift(*R.dense_, s)};
    }
    return {shift(*R.sparse_, s)};
}

/// @brief Solve \f$(sI - A)x = b\f$. `x` may be `b`.
inline void solve(const auto_shifted_lu &F, const vec<cplx> &b, vec<cplx> &x) {
    std::visit([&](const auto &factor) { solve(factor, b, x); }, F.factor);
}

} // namespace num
