/// @file linear/solvers/hessenberg_resolvent.cpp
/// @brief O(n^2)-per-shift Hessenberg resolvent solver implementation.
#include "kernel/factor.hpp"
#include "linear/solvers/hessenberg_resolvent.hpp"
#include "linear/factorization/hessenberg.hpp"
#include <cmath>
#include <stdexcept>
#include <utility>

namespace num {

HessenbergResolventSolver::HessenbergResolventSolver(const Matrix &A) : decomp_(A) {
    debug::check_dim(A.rows(), A.cols(), "HessenbergResolventSolver matrix must be square");
    debug::check_non_empty(A.rows(), "HessenbergResolventSolver matrix");
}

HessenbergResolventSolver::HessenbergResolventSolver(HessenbergDecomposition decomp)
    : decomp_(std::move(decomp)) {}

std::vector<cplx> HessenbergResolventSolver::solve(cplx shift, const Vector &b) const {
    debug::check_dim(decomp_.size(), b.size(), "HessenbergResolventSolver RHS");
    const idx n = decomp_.size();
    const auto b_tilde = hessenberg_project(decomp_.Q(), b);

    std::vector<cplx> y(n);
    std::vector<cplx> M_buf(n * n);
    std::vector<idx> pivots(n);
    hessenberg_shifted_solve(decomp_.H(), shift, b_tilde, y, M_buf, pivots);

    std::vector<cplx> x(n);
    hessenberg_back_project(decomp_.Q(), y, x);
    return x;
}

std::vector<cplx> HessenbergResolventSolver::solve(cplx shift,
                                                  const std::vector<cplx> &b) const {
    debug::check_dim(decomp_.size(), static_cast<idx>(b.size()), "HessenbergResolventSolver RHS");
    const idx n = decomp_.size();
    const auto b_tilde = hessenberg_project(decomp_.Q(), b);

    std::vector<cplx> y(n);
    std::vector<cplx> M_buf(n * n);
    std::vector<idx> pivots(n);
    hessenberg_shifted_solve(decomp_.H(), shift, b_tilde, y, M_buf, pivots);

    std::vector<cplx> x(n);
    hessenberg_back_project(decomp_.Q(), y, x);
    return x;
}

std::vector<std::vector<cplx>>
HessenbergResolventSolver::solve_batch(const std::vector<cplx> &shifts, const Vector &b) const {
    debug::check_dim(decomp_.size(), b.size(), "HessenbergResolventSolver RHS");
    const idx n = decomp_.size();
    const auto b_tilde = hessenberg_project(decomp_.Q(), b);

    std::vector<std::vector<cplx>> results(shifts.size(), std::vector<cplx>(n));

#if defined(_OPENMP)
#pragma omp parallel if (shifts.size() > 2)
#endif
    {
        std::vector<cplx> y(n);
        std::vector<cplx> M_buf(n * n);
        std::vector<idx> pivots(n);

#if defined(_OPENMP)
#pragma omp for
#endif
        for (std::size_t k = 0; k < shifts.size(); ++k) {
            hessenberg_shifted_solve(decomp_.H(), shifts[k], b_tilde, y, M_buf, pivots);
            hessenberg_back_project(decomp_.Q(), y, results[k]);
        }
    }

    return results;
}

std::vector<std::vector<std::vector<cplx>>>
HessenbergResolventSolver::solve_batch(const std::vector<cplx> &shifts,
                                      const std::vector<Vector> &rhs_list) const {
    const idx n = decomp_.size();
    const std::size_t num_rhs = rhs_list.size();
    std::vector<std::vector<cplx>> b_tilde_list(num_rhs);
    for (std::size_t r = 0; r < num_rhs; ++r) {
        debug::check_dim(n, rhs_list[r].size(), "HessenbergResolventSolver RHS list");
        b_tilde_list[r] = hessenberg_project(decomp_.Q(), rhs_list[r]);
    }

    std::vector<std::vector<std::vector<cplx>>> results(
        shifts.size(), std::vector<std::vector<cplx>>(num_rhs, std::vector<cplx>(n)));

#if defined(_OPENMP)
#pragma omp parallel if (shifts.size() > 2)
#endif
    {
        std::vector<cplx> y(n);
        std::vector<cplx> M_buf(n * n);
        std::vector<idx> pivots(n);

#if defined(_OPENMP)
#pragma omp for collapse(2)
#endif
        for (std::size_t k = 0; k < shifts.size(); ++k) {
            for (std::size_t r = 0; r < num_rhs; ++r) {
                hessenberg_shifted_solve(decomp_.H(), shifts[k], b_tilde_list[r], y, M_buf, pivots);
                hessenberg_back_project(decomp_.Q(), y, results[k][r]);
            }
        }
    }

    return results;
}

} // namespace num
