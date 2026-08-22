/// @file src/linalg/solvers/hessenberg_resolvent.cpp
/// @brief O(n^2)-per-shift Hessenberg resolvent solver implementation.
#include "linalg/solvers/hessenberg_resolvent.hpp"
#include <cmath>
#include <stdexcept>
#include <utility>

namespace num {

namespace {

/// Solve (sI - H) y = b_tilde in O(n^2) where H is upper Hessenberg.
void solve_hessenberg_system(const Matrix &H, cplx shift, const std::vector<cplx> &b_tilde,
                             std::vector<cplx> &y, std::vector<cplx> &M_buf,
                             std::vector<idx> &pivots) {
    const idx n = H.rows();
    if (b_tilde.size() != n) {
        throw std::invalid_argument("solve_hessenberg_system: dimension mismatch");
    }

    if (M_buf.size() < n * n) {
        M_buf.resize(n * n);
    }
    if (pivots.size() < n) {
        pivots.resize(n);
    }
    if (y.size() != n) {
        y.resize(n);
    }

    // 1. Form M = sI - H in M_buf
    for (idx i = 0; i < n; ++i) {
        for (idx j = 0; j < n; ++j) {
            M_buf[(i * n) + j] = (i == j ? shift : cplx(0.0, 0.0)) - H(i, j);
        }
    }

    // 2. Hessenberg Gaussian elimination with partial pivoting in O(n^2)
    for (idx i = 0; i + 1 < n; ++i) {
        const double diag_abs = std::abs(M_buf[(i * n) + i]);
        const double subdiag_abs = std::abs(M_buf[((i + 1) * n) + i]);

        if (subdiag_abs > diag_abs) {
            // Swap row i and row i+1 from column i onwards
            for (idx j = i; j < n; ++j) {
                std::swap(M_buf[(i * n) + j], M_buf[((i + 1) * n) + j]);
            }
            pivots[i] = i + 1;
        } else {
            pivots[i] = i;
        }

        const cplx pivot_val = M_buf[(i * n) + i];
        if (std::abs(pivot_val) > 1e-30) {
            const cplx mult = M_buf[((i + 1) * n) + i] / pivot_val;
            M_buf[((i + 1) * n) + i] = mult;
            for (idx j = i + 1; j < n; ++j) {
                M_buf[((i + 1) * n) + j] -= mult * M_buf[(i * n) + j];
            }
        }
    }

    // 3. Forward substitution on RHS
    for (idx i = 0; i < n; ++i) {
        y[i] = b_tilde[i];
    }
    for (idx i = 0; i + 1 < n; ++i) {
        if (pivots[i] != i) {
            std::swap(y[i], y[i + 1]);
        }
        const cplx mult = M_buf[((i + 1) * n) + i];
        y[i + 1] -= mult * y[i];
    }

    // 4. Backward substitution on upper triangular factor
    for (idx step = 0; step < n; ++step) {
        const idx i = n - 1 - step;
        cplx sum = y[i];
        for (idx j = i + 1; j < n; ++j) {
            sum -= M_buf[(i * n) + j] * y[j];
        }
        const cplx diag = M_buf[(i * n) + i];
        if (std::abs(diag) < 1e-30) {
            y[i] = cplx(0.0, 0.0);
        } else {
            y[i] = sum / diag;
        }
    }
}

/// Compute b_tilde = Q^T * b in O(n^2) for real vector b.
std::vector<cplx> project_rhs(const Matrix &Q, const Vector &b) {
    const idx n = Q.rows();
    std::vector<cplx> b_tilde(n, cplx(0.0, 0.0));
    for (idx i = 0; i < n; ++i) {
        double sum = 0.0;
        for (idx j = 0; j < n; ++j) {
            sum += Q(j, i) * b[j]; // Q^T(i, j) = Q(j, i)
        }
        b_tilde[i] = sum;
    }
    return b_tilde;
}

/// Compute b_tilde = Q^T * b in O(n^2) for complex vector b.
std::vector<cplx> project_rhs(const Matrix &Q, const std::vector<cplx> &b) {
    const idx n = Q.rows();
    std::vector<cplx> b_tilde(n, cplx(0.0, 0.0));
    for (idx i = 0; i < n; ++i) {
        cplx sum(0.0, 0.0);
        for (idx j = 0; j < n; ++j) {
            sum += Q(j, i) * b[j];
        }
        b_tilde[i] = sum;
    }
    return b_tilde;
}

/// Compute x = Q * y in O(n^2).
void back_project(const Matrix &Q, const std::vector<cplx> &y, std::vector<cplx> &x) {
    const idx n = Q.rows();
    if (x.size() != n) {
        x.resize(n);
    }
    for (idx i = 0; i < n; ++i) {
        cplx sum(0.0, 0.0);
        for (idx j = 0; j < n; ++j) {
            sum += Q(i, j) * y[j];
        }
        x[i] = sum;
    }
}

} // namespace

HessenbergResolventSolver::HessenbergResolventSolver(const Matrix &A) : decomp_(A) {
    debug::check_dim(A.rows(), A.cols(), "HessenbergResolventSolver matrix must be square");
    debug::check_non_empty(A.rows(), "HessenbergResolventSolver matrix");
}

HessenbergResolventSolver::HessenbergResolventSolver(HessenbergDecomposition decomp)
    : decomp_(std::move(decomp)) {}

std::vector<cplx> HessenbergResolventSolver::solve(cplx shift, const Vector &b) const {
    debug::check_dim(decomp_.size(), b.size(), "HessenbergResolventSolver RHS");
    const idx n = decomp_.size();
    const auto b_tilde = project_rhs(decomp_.Q(), b);

    std::vector<cplx> y(n);
    std::vector<cplx> M_buf(n * n);
    std::vector<idx> pivots(n);
    solve_hessenberg_system(decomp_.H(), shift, b_tilde, y, M_buf, pivots);

    std::vector<cplx> x(n);
    back_project(decomp_.Q(), y, x);
    return x;
}

std::vector<cplx> HessenbergResolventSolver::solve(cplx shift,
                                                  const std::vector<cplx> &b) const {
    debug::check_dim(decomp_.size(), static_cast<idx>(b.size()), "HessenbergResolventSolver RHS");
    const idx n = decomp_.size();
    const auto b_tilde = project_rhs(decomp_.Q(), b);

    std::vector<cplx> y(n);
    std::vector<cplx> M_buf(n * n);
    std::vector<idx> pivots(n);
    solve_hessenberg_system(decomp_.H(), shift, b_tilde, y, M_buf, pivots);

    std::vector<cplx> x(n);
    back_project(decomp_.Q(), y, x);
    return x;
}

std::vector<std::vector<cplx>>
HessenbergResolventSolver::solve_batch(const std::vector<cplx> &shifts, const Vector &b) const {
    debug::check_dim(decomp_.size(), b.size(), "HessenbergResolventSolver RHS");
    const idx n = decomp_.size();
    const auto b_tilde = project_rhs(decomp_.Q(), b);

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
            solve_hessenberg_system(decomp_.H(), shifts[k], b_tilde, y, M_buf, pivots);
            back_project(decomp_.Q(), y, results[k]);
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
        b_tilde_list[r] = project_rhs(decomp_.Q(), rhs_list[r]);
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
                solve_hessenberg_system(decomp_.H(), shifts[k], b_tilde_list[r], y, M_buf, pivots);
                back_project(decomp_.Q(), y, results[k][r]);
            }
        }
    }

    return results;
}

} // namespace num
