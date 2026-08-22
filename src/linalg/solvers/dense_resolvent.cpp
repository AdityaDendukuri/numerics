#include "linalg/solvers/dense_resolvent.hpp"
#include "core/debug.hpp"
#include "linalg/factorization/hessenberg.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace num {

namespace {

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

    const double *H_data = H.data();

    // 1. Form M = sI - H in M_buf
    for (idx i = 0; i < n; ++i) {
        const double *H_row = &H_data[i * n];
        cplx *M_row = &M_buf[i * n];
        for (idx j = 0; j < n; ++j) {
            M_row[j] = (i == j ? shift : cplx(0.0, 0.0)) - H_row[j];
        }
    }

    // 2. Hessenberg Gaussian elimination with partial pivoting in O(n^2)
    for (idx i = 0; i + 1 < n; ++i) {
        cplx *row_i = &M_buf[i * n];
        cplx *row_next = &M_buf[(i + 1) * n];

        const double diag_abs = std::abs(row_i[i]);
        const double subdiag_abs = std::abs(row_next[i]);

        if (subdiag_abs > diag_abs) {
            for (idx j = i; j < n; ++j) {
                std::swap(row_i[j], row_next[j]);
            }
            pivots[i] = i + 1;
        } else {
            pivots[i] = i;
        }

        const cplx pivot_val = row_i[i];
        if (std::abs(pivot_val) > 1e-30) {
            const cplx mult = row_next[i] / pivot_val;
            row_next[i] = mult;
            for (idx j = i + 1; j < n; ++j) {
                row_next[j] -= mult * row_i[j];
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
        const cplx *row_i = &M_buf[i * n];
        cplx sum = y[i];
        for (idx j = i + 1; j < n; ++j) {
            sum -= row_i[j] * y[j];
        }
        const cplx diag = row_i[i];
        if (std::abs(diag) < 1e-30) {
            y[i] = cplx(0.0, 0.0);
        } else {
            y[i] = sum / diag;
        }
    }
}

void project_rhs(const Matrix &Q, const std::vector<cplx> &b, std::vector<cplx> &b_tilde) {
    const idx n = Q.rows();
    if (b_tilde.size() != n) {
        b_tilde.resize(n);
    }
    for (idx i = 0; i < n; ++i) {
        b_tilde[i] = cplx(0.0, 0.0);
    }
    const double *Q_data = Q.data();
    for (idx j = 0; j < n; ++j) {
        const double *row_j = &Q_data[j * n];
        const cplx bj = b[j];
        for (idx i = 0; i < n; ++i) {
            b_tilde[i] += row_j[i] * bj;
        }
    }
}

void back_project(const Matrix &Q, const std::vector<cplx> &y, std::vector<cplx> &x) {
    const idx n = Q.rows();
    if (x.size() != n) {
        x.resize(n);
    }
    const double *Q_data = Q.data();
    for (idx i = 0; i < n; ++i) {
        const double *row_i = &Q_data[i * n];
        cplx sum(0.0, 0.0);
        for (idx j = 0; j < n; ++j) {
            sum += row_i[j] * y[j];
        }
        x[i] = sum;
    }
}

} // namespace

struct DenseResolventSolver::Impl {
    explicit Impl(Matrix input)
        : decomp(input), M_buf(input.rows() * input.rows()), pivots(input.rows()) {
        debug::check_dim(input.rows(), input.cols(), "DenseResolventSolver requires a square matrix");
        debug::check_non_empty(input.rows(), "DenseResolventSolver matrix");
    }

    HessenbergDecomposition decomp;
    cplx current_shift{0.0, 0.0};
    std::vector<cplx> M_buf;
    std::vector<idx> pivots;
    bool factored = false;
};

DenseResolventSolver::DenseResolventSolver(const Matrix &matrix)
    : impl_(std::make_unique<Impl>(matrix)) {}

DenseResolventSolver::DenseResolventSolver(const SparseMatrix &matrix)
    : impl_(std::make_unique<Impl>(dense(matrix))) {}

DenseResolventSolver::~DenseResolventSolver() = default;
DenseResolventSolver::DenseResolventSolver(DenseResolventSolver &&) noexcept = default;
DenseResolventSolver &DenseResolventSolver::operator=(DenseResolventSolver &&) noexcept = default;

idx DenseResolventSolver::size() const noexcept {
    return impl_ ? impl_->decomp.size() : 0;
}

void DenseResolventSolver::factorize(cplx shift) {
    const idx n = impl_->decomp.size();
    impl_->current_shift = shift;

    // 1. Form M = sI - H in M_buf
    const auto &H = impl_->decomp.H();
    const double *H_data = H.data();
    for (idx i = 0; i < n; ++i) {
        const double *H_row = &H_data[i * n];
        cplx *M_row = &impl_->M_buf[i * n];
        for (idx j = 0; j < n; ++j) {
            M_row[j] = (i == j ? shift : cplx(0.0, 0.0)) - H_row[j];
        }
    }

    // 2. Hessenberg Gaussian elimination with partial pivoting in O(n^2)
    for (idx i = 0; i + 1 < n; ++i) {
        cplx *row_i = &impl_->M_buf[i * n];
        cplx *row_next = &impl_->M_buf[(i + 1) * n];

        const double diag_abs = std::abs(row_i[i]);
        const double subdiag_abs = std::abs(row_next[i]);

        if (subdiag_abs > diag_abs) {
            for (idx j = i; j < n; ++j) {
                std::swap(row_i[j], row_next[j]);
            }
            impl_->pivots[i] = i + 1;
        } else {
            impl_->pivots[i] = i;
        }

        const cplx pivot_val = row_i[i];
        if (std::abs(pivot_val) > 1e-30) {
            const cplx mult = row_next[i] / pivot_val;
            row_next[i] = mult;
            for (idx j = i + 1; j < n; ++j) {
                row_next[j] -= mult * row_i[j];
            }
        }
    }

    impl_->factored = true;
}

std::vector<cplx> DenseResolventSolver::solve(const std::vector<cplx> &rhs) const {
    std::vector<cplx> result;
    solve(rhs, result);
    return result;
}

void DenseResolventSolver::solve(const std::vector<cplx> &rhs, std::vector<cplx> &result) const {
    const idx n = impl_->decomp.size();
    if (!impl_->factored) {
        throw std::invalid_argument("DenseResolventSolver: factorization required before solve");
    }
    debug::check_dim(n, static_cast<idx>(rhs.size()), "DenseResolventSolver RHS");

    // 1. Project RHS to Hessenberg coordinates: b_tilde = Q^T * rhs (O(n^2))
    std::vector<cplx> b_tilde(n);
    project_rhs(impl_->decomp.Q(), rhs, b_tilde);

    // 2. Forward substitution on b_tilde (O(n))
    std::vector<cplx> y = b_tilde;
    for (idx i = 0; i + 1 < n; ++i) {
        if (impl_->pivots[i] != i) {
            std::swap(y[i], y[i + 1]);
        }
        const cplx mult = impl_->M_buf[((i + 1) * n) + i];
        y[i + 1] -= mult * y[i];
    }

    // 3. Backward substitution on upper triangular factor (O(n^2))
    for (idx step = 0; step < n; ++step) {
        const idx i = n - 1 - step;
        const cplx *row_i = &impl_->M_buf[i * n];
        cplx sum = y[i];
        for (idx j = i + 1; j < n; ++j) {
            sum -= row_i[j] * y[j];
        }
        const cplx diag = row_i[i];
        if (std::abs(diag) < 1e-30) {
            y[i] = cplx(0.0, 0.0);
        } else {
            y[i] = sum / diag;
        }
    }

    // 4. Back-project solution: result = Q * y (O(n^2))
    back_project(impl_->decomp.Q(), y, result);
}

std::vector<std::vector<cplx>>
DenseResolventSolver::solve(const std::vector<std::vector<cplx>> &right_hand_sides) const {
    std::vector<std::vector<cplx>> result(right_hand_sides.size());
    for (idx index = 0; index < right_hand_sides.size(); ++index) {
        solve(right_hand_sides[index], result[index]);
    }
    return result;
}

} // namespace num
