/// @file linear/factorization/lu.hpp
/// @brief LU factorization, pivoted or unpivoted, in any precision.
#pragma once

#include "container/matrix.hpp"
#include "core/debug.hpp"
#include "core/policy.hpp"
#include "core/types.hpp"
#include "kernel/dense.hpp"
#include "kernel/kernel.hpp"
#include "lapack/lapack_wrapper.hpp"
#include "linear/matrix_properties.hpp"
#include "linear/solvers/solver_result.hpp"
#include "operator/concepts.hpp"
#include "linear/matrix_utils.hpp"
#include <algorithm>
#include <cmath>
#include <concepts>
#include <ostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace num {

/// @brief LU factorization, partial pivoting, packed in place: \f$P A = L U\f$.
/// Returns false on a zero pivot column.
template <std::floating_point T, class Index>
[[nodiscard]] inline bool lu_factor(T *NUM_K_RESTRICT LU, Index *NUM_K_RESTRICT piv,
                                    idx n) noexcept {
    bool nonsingular = true;
    for (idx k = 0; k < n; ++k) {
        idx pivot_row = k;
        T best = std::abs(LU[(k * n) + k]);
        for (idx i = k + 1; i < n; ++i) {
            const T candidate = std::abs(LU[(i * n) + k]);
            if (candidate > best) {
                best = candidate;
                pivot_row = i;
            }
        }
        piv[k] = static_cast<Index>(pivot_row);

        if (pivot_row != k) {
            kernel::swap_rows(LU, n, k, pivot_row, n);
        }
        const T pivot = LU[(k * n) + k];
        if (pivot == T(0)) {
            nonsingular = false;
            continue;
        }
        for (idx i = k + 1; i < n; ++i) {
            const T factor = LU[(i * n) + k] / pivot;
            LU[(i * n) + k] = factor;
            for (idx j = k + 1; j < n; ++j) {
                LU[(i * n) + j] -= factor * LU[(k * n) + j];
            }
        }
    }
    return nonsingular;
}

/// @brief Blocked LU factorization, packed in place: \f$P A = L U\f$.
template <std::floating_point T, class Index>
[[nodiscard]] inline bool lu_factor_blocked(T *NUM_K_RESTRICT LU, Index *NUM_K_RESTRICT piv, idx n,
                                            idx block_size = 64) noexcept {
    if (block_size == 0)
        block_size = 1;
    bool nonsingular = true;
    for (idx kk = 0; kk < n; kk += block_size) {
        const idx kb = std::min(block_size, n - kk), panel_end = kk + kb;
        for (idx k = kk; k < panel_end; ++k) {
            idx pivot_row = k;
            T best = std::abs(LU[(k * n) + k]);
            for (idx i = k + 1; i < n; ++i) {
                const T candidate = std::abs(LU[(i * n) + k]);
                if (candidate > best) {
                    best = candidate;
                    pivot_row = i;
                }
            }
            piv[k] = static_cast<Index>(pivot_row);
            if (pivot_row != k)
                kernel::swap_rows(LU, n, k, pivot_row, n);
            const T pivot = LU[(k * n) + k];
            if (pivot == T(0)) {
                nonsingular = false;
                continue;
            }
            for (idx i = k + 1; i < n; ++i) {
                const T factor = LU[(i * n) + k] / pivot;
                LU[(i * n) + k] = factor;
                for (idx j = k + 1; j < panel_end; ++j)
                    LU[(i * n) + j] -= factor * LU[(k * n) + j];
            }
        }
        const idx trailing = n - panel_end;
        if (trailing == 0)
            continue;
        // U12 <- L11^{-1} A12.
        kernel::trsm_unit_lower_inplace(LU + (kk * n) + panel_end, n, LU + (kk * n) + kk, n, kb,
                                        trailing);
        // A22 <- A22 - L21*U12.
        kernel::gemm(LU + (panel_end * n) + panel_end, n, LU + (panel_end * n) + kk, n,
                     LU + (kk * n) + panel_end, n, T(-1), T(1), trailing, trailing, kb);
    }
    return nonsingular;
}

/// @brief LU solve from packed factors: \f$x \leftarrow A^{-1} b\f$.
template <std::floating_point T, class Index>
inline void lu_solve(T *NUM_K_RESTRICT x, const T *NUM_K_RESTRICT LU,
                     const Index *NUM_K_RESTRICT piv, const T *b, idx n) noexcept {
    for (idx i = 0; i < n; ++i) {
        x[i] = b[i];
    }
    for (idx k = 0; k < n; ++k) {
        const idx p = static_cast<idx>(piv[k]);
        if (p != k) {
            const T tmp = x[k];
            x[k] = x[p];
            x[p] = tmp;
        }
    }
    // Unit-lower forward substitution, then upper back substitution.
    for (idx i = 1; i < n; ++i) {
        T sum = x[i];
        for (idx j = 0; j < i; ++j) {
            sum -= LU[(i * n) + j] * x[j];
        }
        x[i] = sum;
    }
    for (idx i = n; i-- > 0;) {
        T sum = x[i];
        for (idx j = i + 1; j < n; ++j) {
            sum -= LU[(i * n) + j] * x[j];
        }
        x[i] = sum / LU[(i * n) + i];
    }
}

/// @brief LU inverse from packed factors: \f$X \leftarrow A^{-1}\f$.
/// Out of place; `work` holds `n` elements.
template <std::floating_point T, class Index>
inline void lu_invert(T *NUM_K_RESTRICT inverse, const T *NUM_K_RESTRICT LU,
                      const Index *NUM_K_RESTRICT piv, idx n, T *NUM_K_RESTRICT work) noexcept {
    for (idx col = 0; col < n; ++col) {
        for (idx i = 0; i < n; ++i)
            work[i] = (i == col) ? T(1) : T(0);
        for (idx k = 0; k < n; ++k) {
            const idx p = static_cast<idx>(piv[k]);
            if (p != k)
                std::swap(work[k], work[p]);
        }
        for (idx i = 1; i < n; ++i) {
            T sum = work[i];
            for (idx j = 0; j < i; ++j)
                sum -= LU[(i * n) + j] * work[j];
            work[i] = sum;
        }
        for (idx i = n; i-- > 0;) {
            T sum = work[i];
            for (idx j = i + 1; j < n; ++j)
                sum -= LU[(i * n) + j] * work[j];
            work[i] = sum / LU[(i * n) + i];
        }
        for (idx i = 0; i < n; ++i) {
            inverse[(i * n) + col] = work[i];
        }
    }
}

/// @brief LU factorization, unpivoted, in place: \f$A = L U\f$.
/// Returns false on a zero pivot.
template <std::floating_point T>
[[nodiscard]] inline bool lu_no_pivot(T *A, idx n) noexcept {
    bool nonsingular = true;
    for (idx k = 0; k < n; ++k) {
        if (A[k * n + k] == T(0)) {
            nonsingular = false;
            continue;
        }
        const T inverse_pivot = T(1) / A[k * n + k];
        for (idx i = k + 1; i < n; ++i) {
            A[i * n + k] *= inverse_pivot;
            const T multiplier = A[i * n + k];
            for (idx j = k + 1; j < n; ++j)
                A[i * n + j] -= multiplier * A[k * n + j];
        }
    }
    return nonsingular;
}

/// @brief Blocked LU factorization, unpivoted, in place: \f$A = L U\f$.
template <std::floating_point T>
[[nodiscard]] inline bool lu_no_pivot_blocked(T *A, idx n, idx block_size = 64) noexcept {
    if (block_size == 0)
        block_size = 1;
    bool nonsingular = true;
    for (idx kk = 0; kk < n; kk += block_size) {
        const idx kb = std::min(block_size, n - kk);
        const idx panel_end = kk + kb;
        for (idx k = kk; k < panel_end; ++k) {
            if (A[k * n + k] == T(0)) {
                nonsingular = false;
                continue;
            }
            const T inverse_pivot = T(1) / A[k * n + k];
            for (idx i = k + 1; i < n; ++i) {
                A[i * n + k] *= inverse_pivot;
                const T multiplier = A[i * n + k];
                for (idx j = k + 1; j < panel_end; ++j)
                    A[i * n + j] -= multiplier * A[k * n + j];
            }
        }
        const idx trailing = n - panel_end;
        if (trailing == 0)
            continue;
        kernel::trsm_unit_lower_inplace(A + (kk * n) + panel_end, n, A + (kk * n) + kk, n, kb,
                                        trailing);
        constexpr idx row_block = 128;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if (trailing >= 2 * row_block)
#endif
        for (idx row = 0; row < trailing; row += row_block) {
            const idx rows = std::min(row_block, trailing - row);
            kernel::gemm(A + ((panel_end + row) * n) + panel_end, n,
                         A + ((panel_end + row) * n) + kk, n, A + (kk * n) + panel_end, n, T(-1),
                         T(1), rows, trailing, kb);
        }
    }
    return nonsingular;
}

/// @brief Unpivoted LU solve, multiple right-hand sides: \f$X \leftarrow A^{-1} X\f$.
template <std::floating_point T>
inline void lu_no_pivot_solve_multiple(T *X, const T *LU, idx n, idx columns) noexcept {
    if (columns == 1) {
        for (idx i = 1; i < n; ++i) {
            X[i] -= kernel::dot(LU + (i * n), X, i);
        }
        for (idx i = n; i-- > 0;) {
            const T *row = LU + (i * n);
            X[i] = (X[i] - kernel::dot(row + i + 1, X + i + 1, n - i - 1)) / row[i];
        }
        return;
    }
    constexpr idx rhs_block = 16;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if (columns >= 2 * rhs_block)
#endif
    for (idx c0 = 0; c0 < columns; c0 += rhs_block) {
        const idx c1 = std::min(columns, c0 + rhs_block);
        for (idx i = 0; i < n; ++i)
            for (idx j = 0; j < i; ++j)
                for (idx c = c0; c < c1; ++c)
                    X[i * columns + c] -= LU[i * n + j] * X[j * columns + c];
        for (idx i = n; i-- > 0;) {
            for (idx j = i + 1; j < n; ++j)
                for (idx c = c0; c < c1; ++c)
                    X[i * columns + c] -= LU[i * n + j] * X[j * columns + c];
            for (idx c = c0; c < c1; ++c)
                X[i * columns + c] /= LU[i * n + i];
        }
    }
}

/// @brief Unpivoted LU transposed solve, multiple right-hand sides: \f$X \leftarrow A^{-T} X\f$.
template <std::floating_point T>
inline void lu_no_pivot_solve_transpose_multiple(T *X, const T *LU, idx n, idx columns) noexcept {
    if (columns == 1) {
        for (idx i = 0; i < n; ++i) {
            const T *NUM_K_RESTRICT row = LU + (i * n);
            X[i] /= row[i];
            const T xi = X[i];
            T *NUM_K_RESTRICT pending = X + i + 1;
            NUM_K_IVDEP
            for (idx j = 0; j < n - i - 1; ++j) {
                pending[j] -= row[i + 1 + j] * xi;
            }
        }
        for (idx i = n; i-- > 1;) {
            const T *NUM_K_RESTRICT row = LU + (i * n);
            const T xi = X[i];
            NUM_K_IVDEP
            for (idx j = 0; j < i; ++j) {
                X[j] -= row[j] * xi;
            }
        }
        return;
    }
    constexpr idx rhs_block = 16;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if (columns >= 2 * rhs_block)
#endif
    for (idx c0 = 0; c0 < columns; c0 += rhs_block) {
        const idx c1 = std::min(columns, c0 + rhs_block);
        for (idx i = 0; i < n; ++i) {
            for (idx j = 0; j < i; ++j)
                for (idx c = c0; c < c1; ++c)
                    X[i * columns + c] -= LU[j * n + i] * X[j * columns + c];
            for (idx c = c0; c < c1; ++c)
                X[i * columns + c] /= LU[i * n + i];
        }
        for (idx i = n; i-- > 0;)
            for (idx j = i + 1; j < n; ++j)
                for (idx c = c0; c < c1; ++c)
                    X[i * columns + c] -= LU[j * n + i] * X[j * columns + c];
    }
}

} // namespace num

namespace num {

/// @brief LU factorization result: \f$P A = L U\f$, packed as LAPACK `getrf`.
/// \f$P\f$ is stored as row swaps; `swaps` is empty when unpivoted.
template <std::floating_point T>
struct lu_result {
    mat<T> LU;
    array<idx> swaps;
    bool singular = false; ///< True when a pivot fell below tolerance.

    [[nodiscard]] idx size() const { return LU.rows(); }

    friend std::ostream &operator<<(std::ostream &os, const lu_result &r) {
        os << "lu_result{ dim: " << r.LU.rows() << "x" << r.LU.cols()
           << ", pivoted: " << (r.swaps.empty() ? "false" : "true")
           << ", singular: " << (r.singular ? "true" : "false") << " }";
        return os;
    }
};

/// @brief Tag type of `num::no_pivot`.
struct no_pivot_structure {};
/// @brief Select unpivoted LU.
inline constexpr no_pivot_structure no_pivot{};

namespace seq {
/// @brief Blocked LU factorization, partial pivoting: \f$P A = L U\f$.
template <std::floating_point T>
inline lu_result<T> lu(const mat<T> &A) {
    constexpr T singular_tol = T(1e-14);
    const idx n = A.rows();
    lu_result<T> f;
    f.LU = A;
    f.swaps.resize(n);
    f.singular = !num::lu_factor_blocked(f.LU.data(), f.swaps.data(), n);
    for (idx k = 0; k < n && !f.singular; ++k) {
        f.singular = std::abs(f.LU(k, k)) < singular_tol;
    }
    return f;
}
} // namespace seq

namespace lapack {
inline lu_result<real> lu(const mat<real> &A) {
#if defined(NUMERICS_HAS_LAPACK)
    const idx n = A.rows();
    lu_result<real> f;
    f.LU = A;
    f.swaps.resize(n);
    f.singular = false;

    array<lapack_int> ipiv(n);
    int info =
        LAPACKE_dgetrf(LAPACK_ROW_MAJOR, static_cast<lapack_int>(n), static_cast<lapack_int>(n),
                       f.LU.data(), static_cast<lapack_int>(n), ipiv.data());
    if (info < 0) {
        throw std::runtime_error("lu (lapack): dgetrf argument error, info=" +
                                 std::to_string(info));
    }
    if (info > 0) {
        f.singular = true;
    }

    for (idx k = 0; k < n; ++k) {
        f.swaps[k] = static_cast<idx>(ipiv[k] - 1);
    }

    return f;
#else
    return seq::lu(A);
#endif
}
} // namespace lapack

/// @brief LU factorization, partial pivoting: \f$P A = L U\f$.
/// Uses LAPACK above `lapack_factor_threshold` in double precision.
/// @throws std::invalid_argument If `A` is not square.
/// @note A singular `A` sets `singular`; the factorization still completes.
template <std::floating_point T>
inline lu_result<T> lu(const mat<T> &A) {
    if (A.rows() != A.cols()) {
        throw std::invalid_argument("lu: matrix must be square");
    }
#if defined(NUMERICS_LAPACK_DEFAULT)
    if constexpr (std::is_same_v<T, real>) {
        if (A.rows() > lapack_factor_threshold) {
            return lapack::lu(A);
        }
    }
#endif
    return seq::lu(A);
}

/// @brief LU factorization, unpivoted: \f$A = L U\f$.
/// @warning Small pivots go unchecked; use only when the structure keeps pivots nonzero,
/// as for diagonally dominant M-matrices.
/// @throws std::invalid_argument If `A` is not square.
template <std::floating_point T>
inline lu_result<T> lu(const mat<T> &A, no_pivot_structure) {
    if (A.rows() != A.cols()) {
        throw std::invalid_argument("lu: matrix must be square");
    }
    lu_result<T> f;
    f.LU = A;
    f.singular = !num::lu_no_pivot_blocked(f.LU.data(), f.LU.rows());
    return f;
}

/// @brief LU shift derivative: \f$\dot{L}, \dot{U}\f$ at \f$s = 0\f$ for \f$A + s I = L_s U_s\f$.
/// Packed like `lu_result`.
/// @throws std::invalid_argument If `f` is singular or pivoted.
template <std::floating_point T>
[[nodiscard]] inline mat<T> lu_diagonal_shift_derivative(const lu_result<T> &f) {
    if (f.singular || f.LU.rows() != f.LU.cols()) {
        throw std::invalid_argument("LU shift derivative requires a nonsingular square factor");
    }
    if (!f.swaps.empty()) {
        throw std::invalid_argument("LU shift derivative requires a no-pivot factorization");
    }

    const idx n = f.size();
    mat<T> derivative(n, n, T(0));
    for (idx k = 0; k < n; ++k) {
        // U_kj = A_kj - sum_{p<k} L_kp U_pj.
        for (idx j = k; j < n; ++j) {
            T value = (j == k) ? T(1) : T(0);
            for (idx p = 0; p < k; ++p) {
                value -= derivative(k, p) * f.LU(p, j) + f.LU(k, p) * derivative(p, j);
            }
            derivative(k, j) = value;
        }

        // L_ik = (A_ik - sum_{p<k} L_ip U_pk) / U_kk.
        for (idx i = k + 1; i < n; ++i) {
            T value = T(0);
            for (idx p = 0; p < k; ++p) {
                value -= derivative(i, p) * f.LU(p, k) + f.LU(i, p) * derivative(p, k);
            }
            value -= f.LU(i, k) * derivative(k, k);
            derivative(i, k) = value / f.LU(k, k);
        }
    }
    return derivative;
}

namespace detail {

// X <- P X, applying the swaps in the order the factorization made them.
template <std::floating_point T>
inline void apply_swaps(const lu_result<T> &f, T *X, idx nrhs) {
    for (idx k = 0; k < f.swaps.size(); ++k) {
        if (f.swaps[k] != k) {
            kernel::swap_rows(X, nrhs, k, f.swaps[k], nrhs);
        }
    }
}

// X <- P^T X, undoing the swaps in reverse order.
template <std::floating_point T>
inline void undo_swaps(const lu_result<T> &f, T *X, idx nrhs) {
    for (idx k = f.swaps.size(); k-- > 0;) {
        if (f.swaps[k] != k) {
            kernel::swap_rows(X, nrhs, k, f.swaps[k], nrhs);
        }
    }
}

template <std::floating_point T>
inline void check_rhs(const lu_result<T> &f, idx rows) {
    if (f.LU.cols() != f.size() || rows != f.size()) {
        throw std::invalid_argument("lu solve: dimension mismatch");
    }
}

} // namespace detail

/// @brief LU solve: \f$x \leftarrow A^{-1} b\f$. `x` may alias `b`.
template <std::floating_point T>
inline void solve(const lu_result<T> &f, const std::type_identity_t<vec<T>> &b, vec<T> &x) {
    detail::check_rhs(f, b.size());
    x = b;
    detail::apply_swaps(f, x.data(), 1);
    num::lu_no_pivot_solve_multiple(x.data(), f.LU.data(), f.size(), 1);
}

/// @brief LU solve: \f$X \leftarrow A^{-1} B\f$. `X` may alias `B`.
template <std::floating_point T>
inline void solve(const lu_result<T> &f, const std::type_identity_t<mat<T>> &B, mat<T> &X) {
    detail::check_rhs(f, B.rows());
    X = B;
    const idx n = f.size();
    const idx nrhs = B.cols();
    if (f.swaps.empty()) {
        num::lu_no_pivot_solve_multiple(X.data(), f.LU.data(), n, nrhs);
        return;
    }
    // P B, then L Y = P B and U X = Y, both as blocked triangular solves.
    // Never dgetrs: the kernel solve measured faster at every size, and on a
    // threaded BLAS the LAPACK call pays a thread wake-up per solve.
    detail::apply_swaps(f, X.data(), nrhs);
    kernel::trsm_unit_lower_inplace(X.data(), nrhs, f.LU.data(), n, n, nrhs);
    kernel::trsm_upper_inplace(X.data(), nrhs, f.LU.data(), n, n, nrhs);
}

/// @brief LU transposed solve: \f$x \leftarrow A^{-T} b\f$. `x` may alias `b`.
template <std::floating_point T>
inline void solve_transpose(const lu_result<T> &f, const std::type_identity_t<vec<T>> &b,
                            vec<T> &x) {
    detail::check_rhs(f, b.size());
    x = b;
    num::lu_no_pivot_solve_transpose_multiple(x.data(), f.LU.data(), f.size(), 1);
    detail::undo_swaps(f, x.data(), 1);
}

/// @brief LU transposed solve: \f$X \leftarrow A^{-T} B\f$. `X` may alias `B`.
template <std::floating_point T>
inline void solve_transpose(const lu_result<T> &f, const std::type_identity_t<mat<T>> &B,
                            mat<T> &X) {
    detail::check_rhs(f, B.rows());
    X = B;
    const idx n = f.size();
    const idx nrhs = B.cols();
    if (f.swaps.empty()) {
        num::lu_no_pivot_solve_transpose_multiple(X.data(), f.LU.data(), n, nrhs);
        return;
    }
    kernel::trsm_upper_transpose_inplace(X.data(), nrhs, f.LU.data(), n, n, nrhs);
    kernel::trsm_unit_lower_transpose_inplace(X.data(), nrhs, f.LU.data(), n, n, nrhs);
    detail::undo_swaps(f, X.data(), nrhs);
}

/// @brief Determinant: \f$\det A = \det(P)^{-1} \prod_i u_{ii}\f$.
template <std::floating_point T>
inline T det(const lu_result<T> &f) {
    T product = T(1);
    for (idx i = 0; i < f.size(); ++i) {
        product *= f.LU(i, i);
    }
    idx exchanges = 0;
    for (idx k = 0; k < f.swaps.size(); ++k) {
        if (f.swaps[k] != k) {
            ++exchanges;
        }
    }
    return (exchanges % 2 == 0) ? product : -product;
}

namespace detail {
/// @brief Row swaps of an LU factor, identity when unpivoted.
template <std::floating_point T>
inline array<idx> lu_swaps(const lu_result<T> &f) {
    array<idx> swaps = f.swaps;
    if (swaps.empty()) {
        swaps.resize(f.size());
        for (idx k = 0; k < f.size(); ++k) {
            swaps[k] = k;
        }
    }
    return swaps;
}
} // namespace detail

namespace seq {
/// @brief LU inverse, in-tree: \f$A^{-1}\f$.
template <std::floating_point T>
inline mat<T> inverse(const lu_result<T> &f) {
    const idx n = f.size();
    const array<idx> swaps = num::detail::lu_swaps(f);
    mat<T> inv = f.LU;
    array<T> work(n);
    num::lu_invert(inv.data(), f.LU.data(), swaps.data(), n, work.data());
    return inv;
}
} // namespace seq

namespace lapack {
/// @brief LU inverse through LAPACK `dgetri`: \f$A^{-1}\f$.
inline mat<real> inverse(const lu_result<real> &f) {
#if defined(NUMERICS_HAS_LAPACK)
    const idx n = f.size();
    const array<idx> swaps = num::detail::lu_swaps(f);
    array<lapack_int> ipiv(n);
    for (idx i = 0; i < n; ++i) {
        ipiv[i] = static_cast<lapack_int>(swaps[i] + 1);
    }
    mat<real> inv = f.LU;
    LAPACKE_dgetri(LAPACK_ROW_MAJOR, static_cast<lapack_int>(n), inv.data(),
                   static_cast<lapack_int>(n), ipiv.data());
    return inv;
#else
    return seq::inverse(f);
#endif
}

/// @brief LU solve through LAPACK `dgetrs`: \f$X \leftarrow A^{-1} B\f$. `X` may alias `B`.
inline void solve(const lu_result<real> &f, const mat<real> &B, mat<real> &X) {
#if defined(NUMERICS_HAS_LAPACK)
    const idx n = f.size();
    if (B.rows() != n) {
        throw std::invalid_argument("lu solve: dimension mismatch");
    }
    const array<idx> swaps = num::detail::lu_swaps(f);
    array<lapack_int> ipiv(n);
    for (idx i = 0; i < n; ++i) {
        ipiv[i] = static_cast<lapack_int>(swaps[i] + 1);
    }
    if (&X != &B) {
        X = B;
    }
    LAPACKE_dgetrs(LAPACK_ROW_MAJOR, 'N', static_cast<lapack_int>(n),
                   static_cast<lapack_int>(X.cols()), f.LU.data(), static_cast<lapack_int>(n),
                   ipiv.data(), X.data(), static_cast<lapack_int>(X.cols()));
#else
    num::solve(f, B, X);
#endif
}

/// @brief LU solve through LAPACK `dgetrs`: \f$x \leftarrow A^{-1} b\f$. `x` may alias `b`.
inline void solve(const lu_result<real> &f, const vec<real> &b, vec<real> &x) {
#if defined(NUMERICS_HAS_LAPACK)
    const idx n = f.size();
    if (b.size() != n) {
        throw std::invalid_argument("lu solve: dimension mismatch");
    }
    const array<idx> swaps = num::detail::lu_swaps(f);
    array<lapack_int> ipiv(n);
    for (idx i = 0; i < n; ++i) {
        ipiv[i] = static_cast<lapack_int>(swaps[i] + 1);
    }
    if (&x != &b) {
        x = b;
    }
    LAPACKE_dgetrs(LAPACK_ROW_MAJOR, 'N', static_cast<lapack_int>(n), 1, f.LU.data(),
                   static_cast<lapack_int>(n), ipiv.data(), x.data(), 1);
#else
    num::solve(f, b, x);
#endif
}
} // namespace lapack

/// @brief Inverse: \f$A^{-1}\f$. Uses LAPACK when an optimized LAPACK is available.
template <std::floating_point T>
inline mat<T> inverse(const lu_result<T> &f) {
#if defined(NUMERICS_LAPACK_DEFAULT)
    if constexpr (std::is_same_v<T, real>) {
        return lapack::inverse(f);
    }
#endif
    return seq::inverse(f);
}

/// @brief Unit lower factor \f$L\f$.
template <std::floating_point T>
inline mat<T> lower(const lu_result<T> &f) {
    const idx n = f.size();
    mat<T> L(n, n, T(0));
    for (idx i = 0; i < n; ++i) {
        for (idx j = 0; j < i; ++j) {
            L(i, j) = f.LU(i, j);
        }
        L(i, i) = T(1);
    }
    return L;
}

/// @brief Upper factor \f$U\f$.
template <std::floating_point T>
inline mat<T> upper(const lu_result<T> &f) {
    const idx n = f.size();
    mat<T> U(n, n, T(0));
    for (idx i = 0; i < n; ++i) {
        for (idx j = i; j < n; ++j) {
            U(i, j) = f.LU(i, j);
        }
    }
    return U;
}

} // namespace num
