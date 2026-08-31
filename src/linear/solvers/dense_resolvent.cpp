#include "kernel/factor.hpp"
#include "linear/solvers/dense_resolvent.hpp"
#include "core/debug.hpp"
#include "linear/factorization/hessenberg.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace num {

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
    kernel::raw::hessenberg_shifted_factor(impl_->M_buf.data(), impl_->decomp.H().data(), shift, n,
                                           impl_->pivots.data());
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
    kernel::raw::matvec_transpose_into_complex(b_tilde.data(), impl_->decomp.Q().data(),
                                               rhs.data(), n, n);

    // 2. Substitute through the cached factorization (O(n^2))
    std::vector<cplx> y(n);
    kernel::raw::hessenberg_shifted_substitute(y.data(), impl_->M_buf.data(),
                                               impl_->pivots.data(), b_tilde.data(), n);

    // 4. Back-project solution: result = Q * y (O(n^2))
    hessenberg_back_project(impl_->decomp.Q(), y, result);
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
