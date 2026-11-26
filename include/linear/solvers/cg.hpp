/// @file linear/solvers/cg.hpp
/// @brief Conjugate gradients for SPD systems.
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

/// @brief Result of an iterative method: iterations, residual norm, convergence.
template <std::floating_point T>
struct krylov_result {
    idx iterations = 0;
    T residual = T(0);
    bool converged = false;
};

/// @brief Solve a symmetric positive definite system by conjugate gradients, on raw pointers:
/// \f$A x = b\f$.
/// `A(x, y)` writes \f$y = A x\f$; `work` holds \f$3n\f$. Stops if \f$p^T A p \le 0\f$.
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


} // namespace num

namespace num {

/// @brief Options for conjugate gradients: residual tolerance and iteration limit.
struct cg_options {
    real tolerance = 1e-10;
    idx max_iterations = 1000;
};

/// @brief Solve a symmetric positive definite system by conjugate gradients: \f$A x = b\f$.
/// `x` is the initial guess.
/// @throws std::invalid_argument If dimensions mismatch.
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

/// @brief Solve a system by conjugate gradients without checking that it is SPD: \f$A x = b\f$.
/// @warning On an indefinite or asymmetric matrix the iteration breaks down silently.
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

/// @brief Solve a system by conjugate gradients on the GPU: \f$A x = b\f$.
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
