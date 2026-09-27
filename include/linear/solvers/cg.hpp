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
#include "core/policy.hpp"
#include "kernel/krylov.hpp"
#include "linear/matrix_properties.hpp"
#include "linear/solvers/solver_result.hpp"
#include "operator/concepts.hpp"
#include "linear/math_adapters.hpp"
#include "linear/solvers/math_cg.hpp"
#include <algorithm>
#include <cmath>
#include <span>
#include <stdexcept>

#if defined(NUMERICS_HAS_CUDA)
#include "cuda/cuda_ops.hpp"
#endif

namespace num {

namespace unsafe {

/// @brief Conjugate gradients on a stored matrix, without requiring the SPD invariant.
///
/// On an indefinite or non-symmetric matrix the iteration breaks down silently. Runs through
/// `num::accel`; `num::unsafe::cuda::cg` works on device buffers.
/// @return `solver_result`: `.iterations`, `.residual` (final residual norm), `.converged`.
inline solver_result cg(const mat &A, const vec &b, vec &x, real tol = 1e-10,
                       idx max_iter = 1000) {
    const idx n = b.size();
    if (A.rows() != n || A.cols() != n || x.size() != n) {
        throw std::invalid_argument("Dimension mismatch in cg_method solver");
    }

    // The shared iteration, with the matrix supplied as a matvec. `num::accel`
    // selects how that product is formed; the level-1 work is memory bound and
    // inlines from the kernel.
    vec work(3 * n);
    vec in(n);
    vec out(n);
    const auto apply = [&A, &in, &out, n](const real *src, real *dst) {
        std::copy_n(src, n, in.data());
        matvec(A, in, out);
        std::copy_n(out.data(), n, dst);
    };

    const auto r = kernel::cg(apply, x.data(), b.data(), n, work.data(), tol, max_iter);
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
inline solver_result cg(const mat &A, const vec &b, vec &x, real tol = 1e-10,
                       idx max_iter = 1000) {
    const idx n = b.size();
    if (A.rows() != n || A.cols() != n || x.size() != n) {
        throw std::invalid_argument("Dimension mismatch in cg_method solver");
    }

    const_cast<mat &>(A).to_gpu();
    const_cast<vec &>(b).to_gpu();
    x.to_gpu();

    vec r(n), p(n), Ap(n);
    r.to_gpu();
    p.to_gpu();
    Ap.to_gpu();

    num::cuda::matvec(A.gpu_data(), x.gpu_data(), r.gpu_data(), A.rows(), A.cols());
    num::cuda::scale(r.gpu_data(), n, -1.0);
    num::cuda::axpy(1.0, b.gpu_data(), r.gpu_data(), n);
    num::cuda::to_device(p.gpu_data(), r.gpu_data(), n);

    real rsold = num::cuda::dot(r.gpu_data(), r.gpu_data(), n);
    solver_result result{0, std::sqrt(rsold), false};

    for (idx iter = 0; iter < max_iter; ++iter) {
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

        if (result.residual < tol) {
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

/// @brief Solve \f$A x = b\f$ by conjugate gradients for an operator claiming `law::spd`.
///
/// @param A SPD operator, e.g. `num::assume_spd(A)`.
/// @param b Right-hand side vector.
/// @param x Solution vector (serves as initial guess on input, updated in place).
/// @param tolerance Convergence tolerance on \f$\|b - A x\|_2\f$.
/// @param max_iterations Maximum number of Krylov iterations.
/// @return `solver_result` containing iteration count, final residual norm, and convergence boolean.
/// @throws std::invalid_argument If dimensions of `A`, `b`, and `x` do not match.
/// @see assume_spd, pcg, minres, gmres
template <class Op>
requires math::spd_operator<Op, vec>
inline solver_result cg(const Op &A, const vec &b, vec &x, real tolerance,
                       idx max_iterations = 1000) {
    return cg(A, b, x, cg_options{.tolerance = tolerance, .max_iterations = max_iterations});
}

} // namespace num
