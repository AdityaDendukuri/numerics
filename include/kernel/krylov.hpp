/// @file kernel/krylov.hpp
/// @brief Raw-pointer Krylov solvers: matvec as a callable, no containers, no allocation.
///
/// SPDX-License-Identifier: MIT
/// Part of numerics, (c) 2026 Aditya Dendukuri.
/// https://github.com/AdityaDendukuri/numerics
///
/// Depends only on kernel/vector.hpp. Keep the two attribution lines above with whatever you
/// copy. The operator is a callable `A(const T *x, T *y)` and workspace is caller-supplied.
/// Preconditions are documented, not enforced; inside numerics prefer `num::cg`.
#pragma once

#include "kernel/vector.hpp"
#include <cmath>
#include <concepts>

namespace num::kernel {

/// @brief Iteration count, final residual norm, and convergence flag. `operator<<` lives in
/// `kernel/debug.hpp`, to keep `<ostream>` out of this header.
template <std::floating_point T>
struct krylov_result {
    idx iterations = 0;
    T residual = T(0);
    bool converged = false;
};

/// @brief Conjugate gradients for symmetric positive definite \f$A\f$. Stops if some
/// \f$p^T A p \leq 0\f$.
///
/// @param A        Callable `A(const T *x, T *y)` writing \f$y = Ax\f$.
/// @param x        Solution, used as the initial guess on entry.
/// @param b        Right-hand side, length n.
/// @param n        System dimension.
/// @param work     Caller-supplied scratch of length 3n.
/// @param tol      Absolute tolerance on \f$\|r\|_2\f$.
/// @param max_iter Iteration cap.
/// @return `krylov_result`: `.iterations`, `.residual` (final residual norm), `.converged`.
template <std::floating_point T, class MatVec>
[[nodiscard]] inline krylov_result<T> cg(MatVec &&A, T *NUM_K_RESTRICT x, const T *b, idx n,
                                         T *NUM_K_RESTRICT work, T tol = T(1e-10),
                                         idx max_iter = 1000) {
    T *r = work;
    T *p = work + n;
    T *Ap = work + (2 * n);

    A(x, r);
    for (idx i = 0; i < n; ++i) {
        r[i] = b[i] - r[i];
        p[i] = r[i];
    }

    T rs_old = dot(r, r, n);
    krylov_result<T> result{0, std::sqrt(rs_old), false};
    if (result.residual < tol) {
        result.converged = true;
        return result;
    }

    for (idx iter = 0; iter < max_iter; ++iter) {
        result.iterations = iter + 1;
        A(p, Ap);

        const T pAp = dot(p, Ap, n);
        // Positive definiteness is what makes this quotient a valid step length.
        if (!(pAp > T(0)) || !std::isfinite(pAp)) {
            break;
        }
        const T alpha = rs_old / pAp;

        // x <- x + alpha*p
        axpy(x, p, alpha, n);

        // r <- r - alpha*A*p; rs_new <- r^T*r
        const T rs_new = axpy_norm_sq(r, Ap, -alpha, n);
        result.residual = std::sqrt(rs_new);
        if (result.residual < tol) {
            result.converged = true;
            break;
        }

        const T beta = rs_new / rs_old;
        // p <- r + beta*p
        axpby(p, r, T(1), beta, n);
        rs_old = rs_new;
    }
    return result;
}

/// @brief Preconditioned conjugate gradients. \f$M\f$ must be symmetric positive definite,
/// since PCG is CG in the \f$M^{-1}\f$ inner product.
///
/// @param A        Callable `A(const T *x, T *y)` writing \f$y = Ax\f$.
/// @param M        Callable `M(const T *r, T *z)` writing \f$z \approx M^{-1} r\f$.
/// @param x        Solution, used as the initial guess on entry.
/// @param b        Right-hand side, length n.
/// @param n        System dimension.
/// @param work     Caller-supplied scratch of length 4n.
/// @param tol      Absolute tolerance on \f$\|r\|_2\f$.
/// @param max_iter Iteration cap.
/// @return `krylov_result`: `.iterations`, `.residual` (final residual norm), `.converged`.
template <std::floating_point T, class MatVec, class Precond>
[[nodiscard]] inline krylov_result<T> pcg(MatVec &&A, Precond &&M, T *NUM_K_RESTRICT x, const T *b,
                                          idx n, T *NUM_K_RESTRICT work, T tol = T(1e-10),
                                          idx max_iter = 1000) {
    T *r = work;
    T *z = work + n;
    T *p = work + (2 * n);
    T *Ap = work + (3 * n);

    A(x, r);
    // r <- b - A*x
    axpby(r, b, T(1), T(-1), n);

    krylov_result<T> result{0, std::sqrt(dot(r, r, n)), false};
    if (result.residual < tol) {
        result.converged = true;
        return result;
    }

    M(r, z);
    // p <- M^-1*r
    copy(p, z, n);
    T rz_old = dot(r, z, n);

    for (idx iter = 0; iter < max_iter; ++iter) {
        result.iterations = iter + 1;
        A(p, Ap);

        const T pAp = dot(p, Ap, n);
        if (!(pAp > T(0)) || !std::isfinite(pAp)) {
            break;
        }
        const T alpha = rz_old / pAp;

        // x <- x + alpha*p
        axpy(x, p, alpha, n);

        // r <- r - alpha*A*p; ||r||_2^2
        result.residual = std::sqrt(axpy_norm_sq(r, Ap, -alpha, n));
        if (result.residual < tol) {
            result.converged = true;
            break;
        }

        M(r, z);
        const T rz_new = dot(r, z, n);
        const T beta = rz_new / rz_old;
        // p <- M^-1*r + beta*p
        axpby(p, z, T(1), beta, n);
        rz_old = rz_new;
    }
    return result;
}

} // namespace num::kernel
