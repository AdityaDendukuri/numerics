/// @file linear/solvers/hessenberg_resolvent.cpp
/// @brief O(n^2)-per-shift Hessenberg resolvent implementation.
#include "kernel/complex.hpp"
#include "linear/solvers/hessenberg_resolvent.hpp"
#include "linear/factorization/hessenberg.hpp"
#include <cmath>
#include <stdexcept>
#include <utility>

namespace num {

namespace {

const mat<real> &checked_square(const mat<real> &A) {
    debug::check_dim(A.rows(), A.cols(), "hessenberg_resolvent matrix must be square");
    debug::check_non_empty(A.rows(), "hessenberg_resolvent matrix");
    return A;
}

// Solve in Hessenberg coordinates: x = Q (sI - H)^{-1} Q^T b.
template <class Rhs>
void solve_factored(const hessenberg_shifted_lu &F, const Rhs &b, vec<cplx> &x) {
    const idx n = F.size();
    debug::check_dim(n, b.size(), "hessenberg_resolvent RHS");
    const vec<cplx> b_tilde = hessenberg_project(F.decomp->Q(), b);
    vec<cplx> y(n);
    num::hessenberg_shifted_substitute(y.data(), F.factor.data(), F.pivots.data(), b_tilde.data(),
                                       n);
    hessenberg_back_project(F.decomp->Q(), y, x);
}

} // namespace

hessenberg_resolvent::hessenberg_resolvent(const mat<real> &A)
    : decomp_(std::make_shared<const hessenberg_decomposition>(checked_square(A))) {}

hessenberg_resolvent::hessenberg_resolvent(const spmat &A) : hessenberg_resolvent(dense(A)) {}

hessenberg_resolvent::hessenberg_resolvent(hessenberg_decomposition decomp)
    : decomp_(std::make_shared<const hessenberg_decomposition>(std::move(decomp))) {}

hessenberg_shifted_lu shift(const hessenberg_resolvent &R, cplx s) {
    const idx n = R.size();
    hessenberg_shifted_lu F{R.decomp_, s, array<cplx>(n * n), array<idx>(n)};
    num::hessenberg_shifted_factor(F.factor.data(), F.decomp->H().data(), s, n, F.pivots.data());
    return F;
}

void solve(const hessenberg_shifted_lu &F, const vec<cplx> &b, vec<cplx> &x) {
    solve_factored(F, b, x);
}

void solve(const hessenberg_shifted_lu &F, const vec<real> &b, vec<cplx> &x) {
    solve_factored(F, b, x);
}

vec<cplx> solve(const hessenberg_shifted_lu &F, const vec<real> &b) {
    vec<cplx> x;
    solve_factored(F, b, x);
    return x;
}

array<vec<cplx>> solve_batch(const hessenberg_resolvent &R, const array<cplx> &shifts,
                             const vec<real> &b) {
    const hessenberg_decomposition &decomp = R.decomposition();
    debug::check_dim(decomp.size(), b.size(), "hessenberg_resolvent RHS");
    const idx n = decomp.size();
    const vec<cplx> b_tilde = hessenberg_project(decomp.Q(), b);

    array<vec<cplx>> results(shifts.size());

#if defined(_OPENMP)
#pragma omp parallel if (shifts.size() > 2)
#endif
    {
        vec<cplx> y(n);
        array<cplx> M_buf(n * n);
        array<idx> pivots(n);

#if defined(_OPENMP)
#pragma omp for
#endif
        for (std::size_t k = 0; k < shifts.size(); ++k) {
            hessenberg_shifted_solve(decomp.H(), shifts[k], b_tilde, y, M_buf, pivots);
            hessenberg_back_project(decomp.Q(), y, results[k]);
        }
    }

    return results;
}

array<array<vec<cplx>>> solve_batch(const hessenberg_resolvent &R, const array<cplx> &shifts,
                                    const array<vec<real>> &rhs_list) {
    const hessenberg_decomposition &decomp = R.decomposition();
    const idx n = decomp.size();
    const std::size_t num_rhs = rhs_list.size();
    array<vec<cplx>> b_tilde_list(num_rhs);
    for (std::size_t r = 0; r < num_rhs; ++r) {
        debug::check_dim(n, rhs_list[r].size(), "hessenberg_resolvent RHS list");
        b_tilde_list[r] = hessenberg_project(decomp.Q(), rhs_list[r]);
    }

    array<array<vec<cplx>>> results(shifts.size(), array<vec<cplx>>(num_rhs));

#if defined(_OPENMP)
#pragma omp parallel if (shifts.size() > 2)
#endif
    {
        vec<cplx> y(n);
        array<cplx> M_buf(n * n);
        array<idx> pivots(n);

#if defined(_OPENMP)
#pragma omp for collapse(2)
#endif
        for (std::size_t k = 0; k < shifts.size(); ++k) {
            for (std::size_t r = 0; r < num_rhs; ++r) {
                hessenberg_shifted_solve(decomp.H(), shifts[k], b_tilde_list[r], y, M_buf,
                                         pivots);
                hessenberg_back_project(decomp.Q(), y, results[k][r]);
            }
        }
    }

    return results;
}

} // namespace num
