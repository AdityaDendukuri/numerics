/// @file linear/solvers/hessenberg_resolvent.cpp
/// @brief O(n^2)-per-shift Hessenberg resolvent solver implementation.
#include "kernel/complex.hpp"
#include "kernel/factor.hpp"
#include "linear/solvers/hessenberg_resolvent.hpp"
#include "linear/factorization/hessenberg.hpp"
#include <cmath>
#include <stdexcept>
#include <utility>

namespace num {

hessenberg_resolvent_solver::hessenberg_resolvent_solver(const mat &A) : decomp_(A) {
    debug::check_dim(A.rows(), A.cols(), "hessenberg_resolvent_solver matrix must be square");
    debug::check_non_empty(A.rows(), "hessenberg_resolvent_solver matrix");
}

hessenberg_resolvent_solver::hessenberg_resolvent_solver(hessenberg_decomposition decomp)
    : decomp_(std::move(decomp)) {}

array<cplx> hessenberg_resolvent_solver::solve(cplx shift, const vec &b) const {
    debug::check_dim(decomp_.size(), b.size(), "hessenberg_resolvent_solver RHS");
    const idx n = decomp_.size();
    const auto b_tilde = hessenberg_project(decomp_.Q(), b);

    array<cplx> y(n);
    array<cplx> M_buf(n * n);
    array<idx> pivots(n);
    hessenberg_shifted_solve(decomp_.H(), shift, b_tilde, y, M_buf, pivots);

    array<cplx> x(n);
    hessenberg_back_project(decomp_.Q(), y, x);
    return x;
}

array<cplx> hessenberg_resolvent_solver::solve(cplx shift,
                                                  const array<cplx> &b) const {
    debug::check_dim(decomp_.size(), static_cast<idx>(b.size()), "hessenberg_resolvent_solver RHS");
    const idx n = decomp_.size();
    const auto b_tilde = hessenberg_project(decomp_.Q(), b);

    array<cplx> y(n);
    array<cplx> M_buf(n * n);
    array<idx> pivots(n);
    hessenberg_shifted_solve(decomp_.H(), shift, b_tilde, y, M_buf, pivots);

    array<cplx> x(n);
    hessenberg_back_project(decomp_.Q(), y, x);
    return x;
}

array<array<cplx>>
hessenberg_resolvent_solver::solve_batch(const array<cplx> &shifts, const vec &b) const {
    debug::check_dim(decomp_.size(), b.size(), "hessenberg_resolvent_solver RHS");
    const idx n = decomp_.size();
    const auto b_tilde = hessenberg_project(decomp_.Q(), b);

    array<array<cplx>> results(shifts.size(), array<cplx>(n));

#if defined(_OPENMP)
#pragma omp parallel if (shifts.size() > 2)
#endif
    {
        array<cplx> y(n);
        array<cplx> M_buf(n * n);
        array<idx> pivots(n);

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

array<array<array<cplx>>>
hessenberg_resolvent_solver::solve_batch(const array<cplx> &shifts,
                                      const array<vec> &rhs_list) const {
    const idx n = decomp_.size();
    const std::size_t num_rhs = rhs_list.size();
    array<array<cplx>> b_tilde_list(num_rhs);
    for (std::size_t r = 0; r < num_rhs; ++r) {
        debug::check_dim(n, rhs_list[r].size(), "hessenberg_resolvent_solver RHS list");
        b_tilde_list[r] = hessenberg_project(decomp_.Q(), rhs_list[r]);
    }

    array<array<array<cplx>>> results(
        shifts.size(), array<array<cplx>>(num_rhs, array<cplx>(n)));

#if defined(_OPENMP)
#pragma omp parallel if (shifts.size() > 2)
#endif
    {
        array<cplx> y(n);
        array<cplx> M_buf(n * n);
        array<idx> pivots(n);

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
