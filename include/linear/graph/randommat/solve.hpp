/// @file linear/graph/randommat/solve.hpp
/// @brief Solves with approximate Cholesky factors.
#pragma once

#include "container/vector.hpp"
#include "linear/graph/randommat/types.hpp"
#include <concepts>
#include <cstddef>
#include <vector>

namespace num::randommat {

/// @brief Solve with an approximate Cholesky factor: \f$x \leftarrow P^T C^{-T} C^{-1} P b\f$.
template <typename Float = double, std::integral Index = num::idx>
inline void solve(const cholesky_factor<Float, Index> &factor, const Float *b, Float *x,
                  std::vector<Float> &scratch) {
    const Index n = static_cast<Index>(factor.order.size());
    if (scratch.size() < n) {
        scratch.resize(n);
    }

    // 1. Permute RHS: w = P b
    for (Index k = 0; k < n; ++k) {
        scratch[k] = b[factor.order[k]];
    }

    // 2. Forward solve: L y = w (column-oriented)
    for (Index j = 0; j < n; ++j) {
        const auto &entries = factor.columns[j].entries;
        if (entries.empty()) {
            continue;
        }
        const Float diag = entries[0].value;
        scratch[j] /= diag;
        const Float wj = scratch[j];
        for (std::size_t e = 1; e < entries.size(); ++e) {
            scratch[entries[e].row] -= entries[e].value * wj;
        }
    }

    // 3. Backward solve: L^T v = y (column-oriented)
    for (Index step = 0; step < n; ++step) {
        const Index j = n - 1 - step;
        const auto &entries = factor.columns[j].entries;
        if (entries.empty()) {
            continue;
        }
        Float sum = scratch[j];
        for (std::size_t e = 1; e < entries.size(); ++e) {
            sum -= entries[e].value * scratch[entries[e].row];
        }
        scratch[j] = sum / entries[0].value;
    }

    // 4. Inverse permute: x = P^T v
    for (Index k = 0; k < n; ++k) {
        x[factor.order[k]] = scratch[k];
    }
}

/// @brief Solve with an approximate Cholesky factor, on arrays.
template <typename Float = double, std::integral Index = num::idx>
inline std::vector<Float> solve(const cholesky_factor<Float, Index> &factor,
                                const std::vector<Float> &b) {
    const Index n = static_cast<Index>(factor.order.size());
    std::vector<Float> x(n, 0.0);
    std::vector<Float> scratch(n, 0.0);
    solve(factor, b.data(), x.data(), scratch);
    return x;
}

/// @brief Solve with an approximate Cholesky factor, on `vec<real>`.
template <typename Float = double, std::integral Index = num::idx>
inline void solve(const cholesky_factor<Float, Index> &factor,
                  const vec<Float> &b, vec<Float> &x) {
    const Index n = static_cast<Index>(factor.order.size());
    if (x.size() != n) x = vec<Float>(n, Float{0});
    std::vector<Float> scratch(n, Float{0});
    solve(factor, b.data(), x.data(), scratch);
}

} // namespace num::randommat
