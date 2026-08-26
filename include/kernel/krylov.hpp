/// @file kernel/krylov.hpp
/// @brief Raw-pointer Krylov solvers: matvec as a callable, no containers, no allocation.
///
/// SPDX-License-Identifier: MIT
/// Part of numerics, (c) 2026 Aditya Dendukuri.
/// https://github.com/AdityaDendukuri/numerics
///
/// This file has no dependencies outside the standard library beyond
/// kernel/raw.hpp: copy the two into another project as-is, or lift a single
/// routine. Please keep the two attribution lines above with whatever you take.
///
/// The operator enters as a callable `A(const T *x, T *y)` writing \f$y = Ax\f$, so
/// a consuming project supplies its own matrix type, its own sparse format, or a
/// matrix-free stencil without adapting to any interface here. Workspace is
/// caller-supplied; nothing in this file allocates.
///
/// The invariants these methods require are stated in the documentation rather
/// than enforced, because a raw kernel has no type to carry them. Callers working
/// inside numerics should prefer `num::cg`, which does enforce them.
#pragma once

#include "kernel/raw.hpp"
#include <cmath>
#include <concepts>

namespace num::kernel::raw {

/// @brief Iteration count, final residual norm, and convergence flag.
template <std::floating_point T>
struct KrylovResult {
    idx iterations = 0;
    T residual = T(0);
    bool converged = false;
};

/// @brief Conjugate gradients for symmetric positive definite \f$A\f$.
///
/// Minimizes \f$\tfrac12 x^T A x - b^T x\f$ over the Krylov space
/// \f$\mathcal{K}_k(A, r_0)\f$. Requires \f$A = A^T\f$ and \f$A \succ 0\f$: on an
/// indefinite operator some search direction has \f$p^T A p \leq 0\f$, and the
/// iteration is stopped rather than continued with a meaningless step length.
///
/// @param A        Callable `A(const T *x, T *y)` writing \f$y = Ax\f$.
/// @param x        Solution, used as the initial guess on entry.
/// @param b        Right-hand side, length n.
/// @param n        System dimension.
/// @param work     Caller-supplied scratch of length 3n.
/// @param tol      Absolute tolerance on \f$\|r\|_2\f$.
/// @param max_iter Iteration cap.
template <std::floating_point T, class MatVec>
[[nodiscard]] inline KrylovResult<T> cg(MatVec &&A, T *NUM_K_RESTRICT x, const T *b, idx n,
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
    KrylovResult<T> result{0, std::sqrt(rs_old), false};
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

/// @brief Preconditioned conjugate gradients.
///
/// Applies \f$M^{-1}\f$ once per iteration. \f$M\f$ must itself be symmetric
/// positive definite: PCG is CG in the \f$M^{-1}\f$ inner product, and a
/// non-symmetric preconditioner makes that form non-symmetric, so the short
/// recurrence no longer generates a Krylov basis.
///
/// @param A        Callable `A(const T *x, T *y)` writing \f$y = Ax\f$.
/// @param M        Callable `M(const T *r, T *z)` writing \f$z \approx M^{-1} r\f$.
/// @param x        Solution, used as the initial guess on entry.
/// @param b        Right-hand side, length n.
/// @param n        System dimension.
/// @param work     Caller-supplied scratch of length 4n.
/// @param tol      Absolute tolerance on \f$\|r\|_2\f$.
/// @param max_iter Iteration cap.
template <std::floating_point T, class MatVec, class Precond>
[[nodiscard]] inline KrylovResult<T> pcg(MatVec &&A, Precond &&M, T *NUM_K_RESTRICT x, const T *b,
                                         idx n, T *NUM_K_RESTRICT work, T tol = T(1e-10),
                                         idx max_iter = 1000) {
    T *r = work;
    T *z = work + n;
    T *p = work + (2 * n);
    T *Ap = work + (3 * n);

    A(x, r);
    // r <- b - A*x
    axpby(r, b, T(1), T(-1), n);

    KrylovResult<T> result{0, std::sqrt(dot(r, r, n)), false};
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

} // namespace num::kernel::raw
