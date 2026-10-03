/// @file linear/factorization/mixed_lu.hpp
/// @brief LU in single precision, refined to double-precision accuracy.
#pragma once

#include "container/matrix.hpp"
#include "container/vector.hpp"
#include "kernel/vector.hpp"
#include "linear/condition.hpp"
#include "linear/factorization/lu.hpp"
#include "linear/solve.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>

namespace num {

/// @brief The tag type of `num::mixed_precision`.
struct mixed_precision_structure {};
/// @brief Select LU in `float` with iterative refinement to `double` accuracy.
inline constexpr mixed_precision_structure mixed_precision{};

/// @brief \f$A\f$ factored in single precision, solved to double-precision accuracy.
///
/// The \f$O(n^3)\f$ factorization runs in `float`, where a SIMD register holds twice as many
/// values, so it moves half the memory and does about twice the arithmetic per instruction.
/// Each solve then refines in `double`: with \f$x_0 = \hat{A}^{-1}b\f$ from the `float`
/// factors, repeat \f$r = b - Ax_k\f$ in `double` (\f$O(n^2)\f$) and
/// \f$x_{k+1} = x_k + \hat{A}^{-1}r\f$. Each step shrinks the error by about
/// \f$\kappa(A)\,u_{\text{float}}\f$, so it reaches double accuracy in a few steps whenever
/// \f$\kappa(A)\f$ is well below \f$1/u_{\text{float}} \approx 10^7\f$.
///
/// Construction checks that with `inverse_norm1_estimate` on the `float` factors. When
/// \f$A\f$ is too ill-conditioned it also factors in `double`, and solves use that instead:
/// `refined()` says which.
struct mixed_lu {
    /// \f$\kappa(A)u_{\text{float}}\f$ above which refinement is not trusted to converge.
    static constexpr real max_contraction = 0.1;
    /// Refinement steps before a solve settles for its current iterate, as LAPACK `dsgesv`.
    static constexpr int max_steps = 30;

    mat<real> A;
    lu_result<float> low;
    std::optional<lu_result<real>> fallback;
    real condition_estimate = 0.0; ///< \f$\|A\|_1\f$ times the estimate of \f$\|A^{-1}\|_1\f$.
    real norm_1 = 0.0;   ///< \f$\|A\|_1 = \|A^T\|_\infty\f$, for the transposed stopping test.
    real norm_inf = 0.0; ///< \f$\|A\|_\infty\f$, for the stopping test.

    [[nodiscard]] idx size() const { return A.rows(); }
    /// True when solves refine the `float` factors; false when they use `fallback`.
    [[nodiscard]] bool refined() const { return !fallback.has_value(); }
};

namespace detail {

template <class To, class From>
[[nodiscard]] vec<To> converted(const vec<From> &v) {
    vec<To> result(v.size());
    for (idx i = 0; i < v.size(); ++i) {
        result[i] = static_cast<To>(v[i]);
    }
    return result;
}

template <class To, class From>
[[nodiscard]] mat<To> converted(const mat<From> &M) {
    mat<To> result(M.rows(), M.cols());
    const idx count = M.rows() * M.cols();
    for (idx k = 0; k < count; ++k) {
        result.data()[k] = static_cast<To>(M.data()[k]);
    }
    return result;
}

// r <- b - op(A) x in double, with op(A) = A or A^T.
inline void residual(const mat<real> &A, bool transposed, const vec<real> &b,
                     const vec<real> &x, vec<real> &r) {
    const idx n = A.rows();
    r = b;
    // Both orders walk rows of A, stride 1: a dot product per row for A x, and an axpy per
    // row for A^T x = sum_i x_i A(i, :).
    if (transposed) {
        for (idx i = 0; i < n; ++i) {
            const real xi = x[i];
            const real *row = A.data() + (i * n);
            for (idx j = 0; j < n; ++j) {
                r[j] -= row[j] * xi;
            }
        }
    } else {
        for (idx i = 0; i < n; ++i) {
            r[i] -= kernel::dot(A.data() + (i * n), x.data(), n);
        }
    }
}

// Refine x toward op(A)^{-1} b, correcting with the float factors. Stops, as LAPACK
// `dsgesv`, once ||b - op(A)x||_inf <= sqrt(n) eps ||op(A)||_inf ||x||_inf: the backward
// error a double-precision solve would reach. Testing the residual before correcting means
// a converged iterate costs no further float solve.
inline void refine(const mixed_lu &F, bool transposed, const vec<real> &b, vec<real> &x) {
    auto low_solve = [&](const vec<real> &rhs) {
        vec<float> y;
        if (transposed) {
            solve_transpose(F.low, converted<float>(rhs), y);
        } else {
            solve(F.low, converted<float>(rhs), y);
        }
        return converted<real>(y);
    };
    const real tolerance = std::sqrt(static_cast<real>(F.size())) *
                           std::numeric_limits<real>::epsilon() *
                           (transposed ? F.norm_1 : F.norm_inf);
    vec<real> solution = low_solve(b), r;
    for (int step = 0; step < mixed_lu::max_steps; ++step) {
        residual(F.A, transposed, b, solution, r);
        real residual_norm = 0.0, solution_norm = 0.0;
        for (idx i = 0; i < solution.size(); ++i) {
            residual_norm = std::max(residual_norm, std::abs(r[i]));
            solution_norm = std::max(solution_norm, std::abs(solution[i]));
        }
        if (residual_norm <= tolerance * solution_norm) {
            break;
        }
        const vec<real> correction = low_solve(r);
        for (idx i = 0; i < solution.size(); ++i) {
            solution[i] += correction[i];
        }
    }
    x = std::move(solution);
}

template <bool Transposed>
inline void refine_columns(const mixed_lu &F, const mat<real> &B, mat<real> &X) {
    mat<real> result(B.rows(), B.cols(), 0.0);
    vec<real> column(B.rows()), solution;
    for (idx c = 0; c < B.cols(); ++c) {
        for (idx i = 0; i < B.rows(); ++i) {
            column[i] = B(i, c);
        }
        refine(F, Transposed, column, solution);
        for (idx i = 0; i < B.rows(); ++i) {
            result(i, c) = solution[i];
        }
    }
    X = std::move(result);
}

} // namespace detail

/// @brief Factor \f$A\f$ in `float`, keeping \f$A\f$ in `double` for refinement.
/// @throws std::invalid_argument If `A` is not square.
[[nodiscard]] inline mixed_lu lu(const mat<real> &A, mixed_precision_structure) {
    mixed_lu F{A, lu(detail::converted<float>(A)), std::nullopt, 0.0, 0.0, 0.0};
    const real inverse_norm =
        F.low.singular ? std::numeric_limits<real>::infinity()
                       : static_cast<real>(inverse_norm1_estimate<lu_result<float>, float>(
                             F.low, A.rows()));
    F.norm_1 = opnorm1(A);
    for (idx i = 0; i < A.rows(); ++i) {
        real row = 0.0;
        for (idx j = 0; j < A.cols(); ++j) {
            row += std::abs(A(i, j));
        }
        F.norm_inf = std::max(F.norm_inf, row);
    }
    F.condition_estimate = F.norm_1 * inverse_norm;
    const real contraction = F.condition_estimate * std::numeric_limits<float>::epsilon();
    if (!(contraction <= mixed_lu::max_contraction)) {
        F.fallback = lu(A);
    }
    return F;
}

/// @brief Solve \f$Ax = b\f$ to double-precision accuracy. `x` may be `b`.
inline void solve(const mixed_lu &F, const vec<real> &b, vec<real> &x) {
    if (F.fallback) {
        solve(*F.fallback, b, x);
        return;
    }
    detail::refine(F, false, b, x);
}

/// @brief Solve \f$AX = B\f$, refining each column. `X` may be `B`.
inline void solve(const mixed_lu &F, const mat<real> &B, mat<real> &X) {
    if (F.fallback) {
        solve(*F.fallback, B, X);
        return;
    }
    detail::refine_columns<false>(F, B, X);
}

/// @brief Solve \f$A^Tx = b\f$ to double-precision accuracy. `x` may be `b`.
inline void solve_transpose(const mixed_lu &F, const vec<real> &b, vec<real> &x) {
    if (F.fallback) {
        solve_transpose(*F.fallback, b, x);
        return;
    }
    detail::refine(F, true, b, x);
}

/// @brief Solve \f$A^TX = B\f$, refining each column. `X` may be `B`.
inline void solve_transpose(const mixed_lu &F, const mat<real> &B, mat<real> &X) {
    if (F.fallback) {
        solve_transpose(*F.fallback, B, X);
        return;
    }
    detail::refine_columns<true>(F, B, X);
}

} // namespace num
