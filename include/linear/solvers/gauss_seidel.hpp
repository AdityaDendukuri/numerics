/// @file gauss_seidel.hpp
/// @brief Gauss-Seidel iterative solver
#pragma once

#include "container/matrix.hpp"
#include "container/vector.hpp"
#include "container/vector_ops.hpp"
#include "core/policy.hpp"
#include "linear/solvers/solver_result.hpp"
#include <cmath>
#include <stdexcept>

namespace num {

/// @brief Gauss-Seidel iterative solver for Ax = b.
///
/// Requires strictly diagonally dominant or SPD A; the two are incomparable, so pass
/// `num::assume_diagonally_dominant(A)` or `num::assume_spd(A)`. The sweep is sequential;
/// `Parallel` threads only the residual. Use `num::jacobi<true>` for a parallel sweep.
///
/// @param A        Square matrix
/// @param b        Right-hand side vector
/// @param x        Solution vector (initial guess on input, solution on output)
/// @param tol      Convergence tolerance on residual norm (default 1e-10)
/// @param max_iter Maximum iterations (default 1000)
/// @return solver_result with convergence info
template <class Mat = with_law<mat, law::diagonally_dominant>, bool Parallel = has_omp>
requires claims<Mat, law::diagonally_dominant> || claims<Mat, law::spd>
inline solver_result gauss_seidel(const Mat &A_in, const vec &b, vec &x, real tol = 1e-10,
                                 idx max_iter = 1000) {
    const auto &A = A_in.base();
    constexpr real zero_diag_tol = 1e-15;
    idx n = b.size();
    if (A.rows() != n || A.cols() != n || x.size() != n) {
        throw std::invalid_argument("Dimension mismatch in Gauss-Seidel solver");
    }

    solver_result result{0, 0.0, false};

    for (idx iter = 0; iter < max_iter; ++iter) {
        // Sequential update  -- maintain GS convergence properties
        for (idx i = 0; i < n; ++i) {
            if (std::abs(A(i, i)) < zero_diag_tol) {
                throw std::runtime_error("Zero diagonal in Gauss-Seidel at row " +
                                         std::to_string(i));
            }
            real sigma = 0.0;
            for (idx j = 0; j < n; ++j) {
                if (j != i) {
                    sigma += A(i, j) * x[j];
                }
            }
            x[i] = (b[i] - sigma) / A(i, i);
        }

        // Residual ||b - Ax||
        real res = 0.0;
#if defined(NUMERICS_HAS_OMP)
        if constexpr (Parallel) {
#pragma omp parallel for reduction(+ : res) schedule(static)
            for (idx i = 0; i < n; ++i) {
                real ri = b[i];
                for (idx j = 0; j < n; ++j) {
                    ri -= A(i, j) * x[j];
                }
                res += ri * ri;
            }
        } else
#endif
        {
            for (idx i = 0; i < n; ++i) {
                real ri = b[i];
                for (idx j = 0; j < n; ++j) {
                    ri -= A(i, j) * x[j];
                }
                res += ri * ri;
            }
        }
        result.residual = std::sqrt(res);
        result.iterations = iter + 1;

        if (result.residual < tol) {
            result.converged = true;
            break;
        }
    }
    return result;
}

} // namespace num
