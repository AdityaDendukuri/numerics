/// @file cg.hpp
/// @brief Conjugate gradient solvers.
///
/// Solves \f$Ax=b\f$ for symmetric positive definite \f$A\f$ using
/// \f$\mathcal{K}_k(A,r_0)=\mathrm{span}\{r_0,Ar_0,\ldots,A^{k-1}r_0\}\f$.
#pragma once

#include "container/matrix.hpp"
#include "container/matrix_ops.hpp"
#include "container/vector.hpp"
#include "container/vector_ops.hpp"
#include "core/math/concepts.hpp"
#include "core/policy.hpp"
#include "kernel/kernel.hpp"
#include "linear/math_adapters.hpp"
#include "linear/matrix_properties.hpp"
#include "linear/solvers/solver_result.hpp"
#include "operator/concepts.hpp"
#include <algorithm>
#include <cmath>
#include <concepts>
#include <span>
#include <stdexcept>

#if defined(NUMERICS_HAS_CUDA)
#include "cuda/cuda_ops.hpp"
#endif

namespace num {

/// @brief Iteration count, final residual norm, and convergence flag of the raw-pointer `cg`.
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

    T rs_old = kernel::dot(r, r, n);
    krylov_result<T> result{0, std::sqrt(rs_old), false};
    if (result.residual < tol) {
        result.converged = true;
        return result;
    }

    for (idx iter = 0; iter < max_iter; ++iter) {
        result.iterations = iter + 1;
        A(p, Ap);

        const T pAp = kernel::dot(p, Ap, n);
        // Positive definiteness is what makes this quotient a valid step length.
        if (!(pAp > T(0)) || !std::isfinite(pAp)) {
            break;
        }
        const T alpha = rs_old / pAp;

        // x <- x + alpha*p
        kernel::axpy(x, p, alpha, n);

        // r <- r - alpha*A*p; rs_new <- r^T*r
        const T rs_new = kernel::axpy_norm_sq(r, Ap, -alpha, n);
        result.residual = std::sqrt(rs_new);
        if (result.residual < tol) {
            result.converged = true;
            break;
        }

        const T beta = rs_new / rs_old;
        // p <- r + beta*p
        kernel::axpby(p, r, T(1), beta, n);
        rs_old = rs_new;
    }
    return result;
}

/// @brief Preconditioned conjugate gradients. \f$M\f$ must be symmetric positive definite,
/// since PCG is CG in the \f$M^{-1}\f$ inner product.
///

} // namespace num

namespace num {

/// @brief Convergence options of `num::cg`: an absolute tolerance on the residual norm and an
/// iteration limit.
struct cg_options {
    real tolerance = 1e-10;
    idx max_iterations = 1000;
};

/// @brief Solve \f$A x = b\f$ by conjugate gradients for an operator claiming `law::spd`.
///
/// @param A SPD operator, e.g. `num::assume_spd(A)`.
/// @param b Right-hand side vector.
/// @param x Solution vector (serves as initial guess on input, updated in place).
/// @param options Tolerance on \f$\|b - A x\|_2\f$ and iteration limit.
/// @return `solver_result` containing iteration count, final residual norm, and convergence boolean.
/// @throws std::invalid_argument If dimensions of `A`, `b`, and `x` do not match.
/// @see assume_spd, pcg, minres, gmres
template <class Op, class V>
requires math::inner_product_space<V> &&math::spd_operator<Op, V> &&
        std::floating_point<math::scalar_t<V>> [[nodiscard]] solver_result
        cg(const Op &A, const V &b, V &x, cg_options options = {}) {
    using S = math::scalar_t<V>;

    const auto n = math::dimension(b);
    if (math::dimension(x) != n || A.rows() != n || A.cols() != n) {
        throw std::invalid_argument("cg: incompatible operator and vector dimensions");
    }
    if (!(options.tolerance > 0.0)) {
        throw std::invalid_argument("cg: invalid convergence options");
    }

    V residual = math::zero_like(b);
    V direction = math::zero_like(b);
    V applied = math::zero_like(b);

    math::apply(A, x, residual);
    // r <- b - A*x
    math::linear_combination(S(1), b, S(-1), residual);
    direction = residual;

    S residual_square = math::inner(residual, residual);
    solver_result result{0, static_cast<real>(std::sqrt(residual_square)), false};
    if (result.residual < options.tolerance) {
        result.converged = true;
        return result;
    }

    for (idx iteration = 0; iteration < options.max_iterations; ++iteration) {
        result.iterations = iteration + 1;
        math::apply(A, direction, applied);

        const S curvature = math::inner(direction, applied);
        if (!(curvature > S(0)) || !std::isfinite(curvature)) {
            throw std::runtime_error("cg: positive-definite curvature invariant was violated");
        }

        const S alpha = residual_square / curvature;
        // x <- x + alpha*p
        math::axpy(alpha, direction, x);

        // r <- r - alpha*A*p; ||r||_2^2
        const S next_square = math::axpy_norm_sq(-alpha, applied, residual);
        if (!(next_square >= S(0)) || !std::isfinite(next_square)) {
            throw std::runtime_error("cg: inner-product norm invariant was violated");
        }
        result.residual = static_cast<real>(std::sqrt(next_square));
        if (result.residual < options.tolerance) {
            result.converged = true;
            break;
        }

        const S beta = next_square / residual_square;
        // p <- r + beta*p
        math::linear_combination(S(1), residual, beta, direction);
        residual_square = next_square;
    }

    return result;
}

namespace unsafe {

/// @brief Conjugate gradients on a stored matrix, without requiring the SPD invariant.
///
/// On an indefinite or non-symmetric matrix the iteration breaks down silently. Runs through
/// `num::accel`; `num::unsafe::cuda::cg` works on device buffers.
/// @return `solver_result`: `.iterations`, `.residual` (final residual norm), `.converged`.
inline solver_result cg(const mat<real> &A, const vec<real> &b, vec<real> &x, cg_options options = {}) {
    const idx n = b.size();
    if (A.rows() != n || A.cols() != n || x.size() != n) {
        throw std::invalid_argument("cg: incompatible matrix and vector dimensions");
    }

    // The shared iteration, with the matrix supplied as a matvec. `num::accel`
    // selects how that product is formed; the level-1 work is memory bound and
    // inlines from the kernel.
    vec<real> work(3 * n);
    vec<real> in(n);
    vec<real> out(n);
    const auto apply = [&A, &in, &out, n](const real *src, real *dst) {
        std::copy_n(src, n, in.data());
        matvec(A, in, out);
        std::copy_n(out.data(), n, dst);
    };

    const auto r = num::cg(apply, x.data(), b.data(), n, work.data(), options.tolerance,
                           options.max_iterations);
    return solver_result{r.iterations, r.residual, r.converged};
}

#if defined(NUMERICS_HAS_CUDA)
namespace cuda {

/// @brief Conjugate gradients entirely on the device.
///
/// The device path cannot share the host kernel: its vectors live on the
/// device, so every level-1 operation is a device call rather than a loop over
/// host memory. It therefore keeps the iteration written out, mirroring
/// `num::unsafe::cg`'s structure with `num::cuda::*` in place of `num::accel::*`.
/// @return `solver_result`: `.iterations`, `.residual` (final residual norm), `.converged`.
inline solver_result cg(const mat<real> &A, const vec<real> &b, vec<real> &x, cg_options options = {}) {
    const idx n = b.size();
    if (A.rows() != n || A.cols() != n || x.size() != n) {
        throw std::invalid_argument("cg: incompatible matrix and vector dimensions");
    }

    const_cast<mat<real> &>(A).to_gpu();
    const_cast<vec<real> &>(b).to_gpu();
    x.to_gpu();

    vec<real> r(n), p(n), Ap(n);
    r.to_gpu();
    p.to_gpu();
    Ap.to_gpu();

    num::cuda::matvec(A.gpu_data(), x.gpu_data(), r.gpu_data(), A.rows(), A.cols());
    num::cuda::scale(r.gpu_data(), n, -1.0);
    num::cuda::axpy(1.0, b.gpu_data(), r.gpu_data(), n);
    num::cuda::to_device(p.gpu_data(), r.gpu_data(), n);

    real rsold = num::cuda::dot(r.gpu_data(), r.gpu_data(), n);
    solver_result result{0, std::sqrt(rsold), false};

    for (idx iter = 0; iter < options.max_iterations; ++iter) {
        result.iterations = iter + 1;
        num::cuda::matvec(A.gpu_data(), p.gpu_data(), Ap.gpu_data(), A.rows(), A.cols());

        real pAp = num::cuda::dot(p.gpu_data(), Ap.gpu_data(), n);
        if (!(pAp > 0.0) || !std::isfinite(pAp)) {
            break;
        }
        real alpha = rsold / pAp;

        num::cuda::axpy(alpha, p.gpu_data(), x.gpu_data(), n);
        num::cuda::axpy(-alpha, Ap.gpu_data(), r.gpu_data(), n);

        real rsnew = num::cuda::dot(r.gpu_data(), r.gpu_data(), n);
        result.residual = std::sqrt(rsnew);

        if (result.residual < options.tolerance) {
            result.converged = true;
            break;
        }

        real beta = rsnew / rsold;
        num::cuda::scale(p.gpu_data(), n, beta);
        num::cuda::axpy(1.0, r.gpu_data(), p.gpu_data(), n);
        rsold = rsnew;
    }
    x.to_cpu();
    return result;
}

} // namespace cuda
#endif

} // namespace unsafe

} // namespace num
