/// @file linear/solvers/auto_resolvent.hpp
/// @brief Shifted solves, dense or sparse by size.
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

/// @brief Backend selection: dense up to `dense_limit`, plus a sparse symmetry hint.
struct auto_resolvent_options {
    idx dense_limit = 128;
};

/// @brief Automatic resolvent: \f$(s I - A)^{-1}\f$ by Hessenberg reduction or sparse LU.
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
        sparse_.emplace(A);
    }

    [[nodiscard]] idx size() const noexcept { return dense_ ? dense_->size() : sparse_->size(); }

  private:
    std::optional<hessenberg_resolvent> dense_;
    std::optional<sparse_resolvent> sparse_;
    friend auto_shifted_lu shift(const auto_resolvent &R, cplx s);
};

/// @brief Shifted factor: \f$s I - A\f$ for one \f$s\f$.
struct auto_shifted_lu {
    std::variant<hessenberg_shifted_lu, sparse_shifted_lu> factor;

    [[nodiscard]] idx size() const noexcept {
        return std::visit([](const auto &F) { return F.size(); }, factor);
    }
};

/// @brief Factor \f$s I - A\f$.
[[nodiscard]] inline auto_shifted_lu shift(const auto_resolvent &R, cplx s) {
    if (R.dense_) {
        return {shift(*R.dense_, s)};
    }
    return {shift(*R.sparse_, s)};
}

/// @brief Shifted solve: \f$x \leftarrow (s I - A)^{-1} b\f$. `x` may alias `b`.
inline void solve(const auto_shifted_lu &F, const vec<cplx> &b, vec<cplx> &x) {
    std::visit([&](const auto &factor) { solve(factor, b, x); }, F.factor);
}

} // namespace num
