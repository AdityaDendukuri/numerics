/// @file cg.hpp
/// @brief Conjugate gradient solvers.
///
/// Solves \f$Ax=b\f$ for symmetric positive definite \f$A\f$ using
/// \f$\mathcal{K}_k(A,r_0)=\mathrm{span}\{r_0,Ar_0,\ldots,A^{k-1}r_0\}\f$.
#pragma once

#include "container/matrix_ops.hpp"
#include "container/parallel/cuda_ops.hpp"
#include "container/vector_ops.hpp"
#include "container/matrix.hpp"
#include "core/policy.hpp"
#include "container/vector.hpp"
#include "linear/concepts.hpp"
#include "linear/math_adapters.hpp"
#include "linear/matrix_properties.hpp"
#include "linear/solvers/math_cg.hpp"
#include "linear/solvers/solver_result.hpp"
#include "kernel/krylov.hpp"
#include "operator/concepts.hpp"
#include <algorithm>
#include <cmath>
#include <span>
#include <stdexcept>

namespace num {

namespace unsafe {

/// @brief Conjugate gradients on a stored matrix, without requiring the SPD invariant.
///
/// CG minimizes \f$\tfrac12 x^T A x - b^T x\f$, which is bounded below only when
/// \f$A\f$ is positive definite. On an indefinite matrix the search direction can
/// have \f$p^T A p \leq 0\f$ and the iteration breaks down; on a non-symmetric one
/// the Krylov recurrence is invalid from the first step. Neither is reported as an
/// error, which is why the invariant is normally required.
SolverResult cg(const Matrix &A, const Vector &b, Vector &x, real tol = 1e-10,
                idx max_iter = 1000, Backend backend = backend::dflt);

} // namespace unsafe

/// @brief Solve \f$A x = b\f$ using Conjugate Gradients for certified SPD operators.
///
/// Iteratively minimizes the quadratic form \f$\phi(x) = \frac{1}{2} x^T A x - b^T x\f$ over
/// the Krylov subspace \f$\mathcal{K}_k(A, r_0)\f$. The operator `A` must carry
/// compile-time or runtime positive-definite evidence (`axiom::positive_definite`).
///
/// @tparam Op Linear operator type satisfying `math::EndomorphismOn<Op, Vector>` and carrying SPD evidence.
/// @param A Symmetric positive-definite linear operator or matrix wrapper (e.g. `num::assume_spd(A)`).
/// @param b Right-hand side vector.
/// @param x Solution vector (serves as initial guess on input, updated in place).
/// @param tolerance Convergence tolerance on Euclidean residual norm \f$\|b - A x\|_2\f$.
/// @param max_iterations Maximum number of Krylov iterations before termination.
/// @param backend Optional hardware backend tag (defaults to `backend::dflt`).
/// @return `SolverResult` containing iteration count, final residual norm, and convergence boolean.
/// @throws std::invalid_argument If dimensions of `A`, `b`, and `x` do not match.
/// @see assume_spd, pcg, minres, gmres
template <class Op>
requires math::InnerProductSpace<Vector> && math::EndomorphismOn<Op, Vector> &&
         math::Carries<Op, axiom::positive_definite>
inline SolverResult cg(const Op &A, const Vector &b, Vector &x, real tolerance,
                       idx max_iterations = 1000, Backend = backend::dflt) {
    return cg(A, b, x, CGOptions{.tolerance = tolerance, .max_iterations = max_iterations});
}

namespace unsafe {

inline SolverResult cg(const Matrix &A, const Vector &b, Vector &x, real tol, idx max_iter,
                Backend backend) {
    const idx n = b.size();
    if (A.rows() != n || A.cols() != n || x.size() != n) {
        throw std::invalid_argument("Dimension mismatch in CG solver");
    }

#if defined(NUMERICS_HAS_CUDA)
    if (backend == Backend::gpu) {
        // The device path cannot share the host kernel: its vectors live on the
        // device, so every level-1 operation is a device call rather than a loop
        // over host memory. It therefore keeps the iteration written out.
    // GPU path: transfer all data to device first
        if (backend == Backend::gpu) {
            const_cast<Matrix &>(A).to_gpu();
            const_cast<Vector &>(b).to_gpu();
            x.to_gpu();
        }

        Vector r(n), p(n), Ap(n);
        if (backend == Backend::gpu) {
            r.to_gpu();
            p.to_gpu();
            Ap.to_gpu();
        }

        matvec(A, x, r, backend);
        if (backend == Backend::gpu) {
            scale(r, -1.0, backend);
            axpy(1.0, b, r, backend);
            cuda::to_device(p.gpu_data(), r.gpu_data(), n);
        } else {
            for (idx i = 0; i < n; ++i) {
                r[i] = b[i] - r[i];
            }
            for (idx i = 0; i < n; ++i) {
                p[i] = r[i];
            }
        }

        real rsold = dot(r, r, backend);
        SolverResult result{0, std::sqrt(rsold), false};

        for (idx iter = 0; iter < max_iter; ++iter) {
            result.iterations = iter + 1;
            matvec(A, p, Ap, backend);

            real pAp = dot(p, Ap, backend);
            if (!(pAp > 0.0) || !std::isfinite(pAp)) {
                break;
            }
            real alpha = rsold / pAp;

            axpy(alpha, p, x, backend);
            axpy(-alpha, Ap, r, backend);

            real rsnew = dot(r, r, backend);
            result.residual = std::sqrt(rsnew);

            if (result.residual < tol) {
                result.converged = true;
                break;
            }

            real beta = rsnew / rsold;
            scale(p, beta, backend);
            axpy(1.0, r, p, backend);
            rsold = rsnew;
        }
        x.to_cpu();
        return result;
    }
#endif

    // Everything else is the shared iteration, with the matrix supplied as a
    // matvec. The backend selects how that product is formed; the level-1 work is
    // memory bound and inlines from the kernel.
    Vector work(3 * n);
    Vector in(n);
    Vector out(n);
    const auto apply = [&A, &in, &out, n, backend](const real *src, real *dst) {
        std::copy_n(src, n, in.data());
        matvec(A, in, out, backend);
        std::copy_n(out.data(), n, dst);
    };

    const auto r = kernel::raw::cg(apply, x.data(), b.data(), n, work.data(), tol, max_iter);
    return SolverResult{r.iterations, r.residual, r.converged};
}

} // namespace unsafe

} // namespace num
