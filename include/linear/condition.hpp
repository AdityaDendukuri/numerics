/// @file linear/condition.hpp
/// @brief Condition number estimates from factorizations.
#pragma once

#include "container/matrix.hpp"
#include "container/vector.hpp"
#include "core/types.hpp"
#include "linear/solve.hpp"
#include "linear/sparse/sparse.hpp"
#include <algorithm>
#include <cmath>
#include <concepts>

namespace num {

/// @brief Compute the 1-norm of a matrix: \f$\|A\|_1 = \max_j \sum_i |a_{ij}|\f$.
template <std::floating_point T>
[[nodiscard]] T opnorm1(const mat<T> &A) {
    // Accumulate all column sums in one pass over the rows: A is row-major, so this reads it
    // with stride 1, where summing one column at a time would read it with stride `cols`.
    array<T> column(A.cols(), T(0));
    for (idx i = 0; i < A.rows(); ++i) {
        const T *row = A.data() + (i * A.cols());
        for (idx j = 0; j < A.cols(); ++j) {
            column[j] += std::abs(row[j]);
        }
    }
    return column.empty() ? T(0) : *std::max_element(column.begin(), column.end());
}

/// @brief Compute the 1-norm of a sparse matrix: \f$\|A\|_1\f$.
[[nodiscard]] inline real opnorm1(const spmat &A) {
    array<real> column(A.n_cols(), 0.0);
    for (idx i = 0; i < A.n_rows(); ++i) {
        for (idx k = A.row_ptr()[i]; k < A.row_ptr()[i + 1]; ++k) {
            column[A.col_idx()[k]] += std::abs(A.values()[k]);
        }
    }
    return column.empty() ? 0.0 : *std::max_element(column.begin(), column.end());
}

/// @brief Estimate the 1-norm of the inverse from a factorization (Hager–Higham, as `dlacn2`):
/// \f$\|A^{-1}\|_1\f$.
/// The estimate is a lower bound.
/// @note Usually exact or within a factor of 3, for \f$O(n^2)\f$ work with a dense factor.
template <class F, std::floating_point T = real>
requires requires(const F &factor, const vec<T> &b, vec<T> &x) {
    solve(factor, b, x);
    solve_transpose(factor, b, x);
}
[[nodiscard]] T inverse_norm1_estimate(const F &factor, idx n) {
    if (n == 0) {
        return T(0);
    }
    constexpr int max_steps = 5;
    auto norm1 = [](const vec<T> &v) {
        T total = 0;
        for (idx i = 0; i < v.size(); ++i) {
            total += std::abs(v[i]);
        }
        return total;
    };

    vec<T> x(n, T(1) / static_cast<T>(n)), y, z, sign(n, T(0));
    T estimate = 0;
    idx previous_j = n;
    for (int step = 0; step < max_steps; ++step) {
        solve(factor, x, y);
        const T norm = norm1(y);
        if (step > 0 && norm <= estimate) {
            break; // This vertex is no better than the last one.
        }
        estimate = norm;
        bool same_signs = step > 0;
        for (idx i = 0; i < n; ++i) {
            const T s = y[i] >= T(0) ? T(1) : T(-1);
            same_signs = same_signs && s == sign[i];
            sign[i] = s;
        }
        if (same_signs) {
            break; // The next gradient would repeat this one.
        }
        // z is the gradient of ||A^{-1}x||_1 at x; its largest entry names the best vertex.
        solve_transpose(factor, sign, z);
        idx j = 0;
        for (idx i = 1; i < n; ++i) {
            if (std::abs(z[i]) > std::abs(z[j])) {
                j = i;
            }
        }
        T gradient_at_x = 0;
        for (idx i = 0; i < n; ++i) {
            gradient_at_x += z[i] * x[i];
        }
        if (step > 0 && (j == previous_j || std::abs(z[j]) <= gradient_at_x)) {
            break; // No vertex increases the norm: a local maximum.
        }
        previous_j = j;
        x = vec<T>(n, T(0));
        x[j] = T(1);
    }

    // Higham's extra vector, whose alternating, growing entries catch matrices for which
    // the vertex search stalls.
    for (idx i = 0; i < n; ++i) {
        const T magnitude = n > 1 ? T(1) + (static_cast<T>(i) / static_cast<T>(n - 1)) : T(1);
        x[i] = (i % 2 == 0) ? magnitude : -magnitude;
    }
    solve(factor, x, y);
    const T alternative = T(2) * norm1(y) / (T(3) * static_cast<T>(n));
    return std::max(estimate, alternative);
}

/// @brief Estimate the reciprocal condition number:
/// \f$1 / (\|A\|_1 \|A^{-1}\|_1)\f$; 0 for a zero matrix.
/// @note A solve loses about \f$\log_{10}(1 / \mathrm{rcond})\f$ digits.
template <factorization F>
[[nodiscard]] real rcond(const F &factor, const mat<real> &A) {
    const real norm = opnorm1(A);
    if (norm == 0.0) {
        return 0.0;
    }
    return 1.0 / (norm * inverse_norm1_estimate(factor, A.rows()));
}

/// @brief Estimate the reciprocal condition number of a sparse matrix:
/// \f$1 / (\|A\|_1 \|A^{-1}\|_1)\f$.
template <factorization F>
[[nodiscard]] real rcond(const F &factor, const spmat &A) {
    const real norm = opnorm1(A);
    if (norm == 0.0) {
        return 0.0;
    }
    return 1.0 / (norm * inverse_norm1_estimate(factor, A.n_rows()));
}

} // namespace num
