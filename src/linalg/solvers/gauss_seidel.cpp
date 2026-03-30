#include "linalg/solvers/gauss_seidel.hpp"
#include <cmath>
#include <stdexcept>

namespace num {

// Gauss-Seidel  -- sequential update order, parallel residual when omp backend.
//
// Note: standard Gauss-Seidel has sequential data dependencies (x[i] depends
// on x[0..i-1] updated in the same sweep).  We keep the sequential update
// order here and parallelise only the residual computation.  For a truly
// parallel relaxation scheme, use the Jacobi solver with Backend::omp.
SolverResult gauss_seidel(const Matrix& A, const Vector& b, Vector& x,
                          real tol, idx max_iter, Backend backend) {
    constexpr real zero_diag_tol = 1e-15;
    idx n = b.size();
    if (A.rows() != n || A.cols() != n || x.size() != n)
        throw std::invalid_argument("Dimension mismatch in Gauss-Seidel solver");

    SolverResult result{0, 0.0, false};

    for (idx iter = 0; iter < max_iter; ++iter) {
        // Sequential update  -- maintain GS convergence properties
        for (idx i = 0; i < n; ++i) {
            if (std::abs(A(i, i)) < zero_diag_tol)
                throw std::runtime_error("Zero diagonal in Gauss-Seidel at row " +
                                         std::to_string(i));
            real sigma = 0.0;
            for (idx j = 0; j < n; ++j)
                if (j != i) sigma += A(i, j) * x[j];
            x[i] = (b[i] - sigma) / A(i, i);
        }

        // Residual ||b - Ax||
        real res = 0.0;
#ifdef NUMERICS_HAS_OMP
        #pragma omp parallel for reduction(+:res) schedule(static) if(backend == Backend::omp)
#endif
        for (idx i = 0; i < n; ++i) {
            real ri = b[i];
            for (idx j = 0; j < n; ++j) ri -= A(i, j) * x[j];
            res += ri * ri;
        }
        result.residual   = std::sqrt(res);
        result.iterations = iter + 1;

        if (result.residual < tol) {
            result.converged = true;
            break;
        }
    }
    return result;
}

} // namespace num
