#include "linalg/solvers/cg.hpp"
#include "core/parallel/cuda_ops.hpp"
#include <cmath>
#include <stdexcept>

namespace num {

SolverResult cg(const Matrix& A, const Vector& b, Vector& x,
                real tol, idx max_iter, Backend backend) {
    idx n = b.size();
    if (A.rows() != n || A.cols() != n || x.size() != n)
        throw std::invalid_argument("Dimension mismatch in CG solver");

    // GPU path: transfer all data to device first
    if (backend == Backend::gpu) {
        const_cast<Matrix&>(A).to_gpu();
        const_cast<Vector&>(b).to_gpu();
        x.to_gpu();
    }

    Vector r(n), p(n), Ap(n);
    if (backend == Backend::gpu) { r.to_gpu(); p.to_gpu(); Ap.to_gpu(); }

    matvec(A, x, r, backend);
    if (backend == Backend::gpu) {
        scale(r, -1.0, backend);
        axpy(1.0, b, r, backend);
        cuda::to_device(p.gpu_data(), r.gpu_data(), n);
    } else {
        for (idx i = 0; i < n; ++i) r[i] = b[i] - r[i];
        for (idx i = 0; i < n; ++i) p[i] = r[i];
    }

    real rsold = dot(r, r, backend);
    SolverResult result{0, std::sqrt(rsold), false};

    for (idx iter = 0; iter < max_iter; ++iter) {
        result.iterations = iter + 1;
        matvec(A, p, Ap, backend);

        real pAp = dot(p, Ap, backend);
        if (std::abs(pAp) < 1e-15) break;
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

    if (backend == Backend::gpu) x.to_cpu();
    return result;
}

SolverResult cg_matfree(MatVecFn matvec_fn, const Vector& b, Vector& x,
                        real tol, idx max_iter) {
    idx n = b.size();
    Vector r(n), p(n), Ap(n);

    matvec_fn(x, r);
    for (idx i = 0; i < n; ++i) r[i] = b[i] - r[i];
    for (idx i = 0; i < n; ++i) p[i] = r[i];

    real rsold = dot(r, r, Backend::seq);
    SolverResult result{0, std::sqrt(rsold), false};

    for (idx iter = 0; iter < max_iter; ++iter) {
        result.iterations = iter + 1;
        matvec_fn(p, Ap);

        real pAp = dot(p, Ap, Backend::seq);
        if (std::abs(pAp) < 1e-15) break;
        real alpha = rsold / pAp;

        axpy(alpha, p, x, Backend::seq);
        axpy(-alpha, Ap, r, Backend::seq);

        real rsnew = dot(r, r, Backend::seq);
        result.residual = std::sqrt(rsnew);
        if (result.residual < tol) { result.converged = true; break; }

        real beta = rsnew / rsold;
        scale(p, beta, Backend::seq);
        axpy(1.0, r, p, Backend::seq);
        rsold = rsnew;
    }
    return result;
}

} // namespace num
