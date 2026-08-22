#include "linalg/solvers/auto_resolvent.hpp"
#include "linalg/solvers/dense_resolvent.hpp"
#include <stdexcept>

namespace num {

struct AutoResolventSolver::Impl {
    std::unique_ptr<DenseResolventSolver> dense;
    std::unique_ptr<SparseResolventSolver> sparse;
};

AutoResolventSolver::AutoResolventSolver(const SparseMatrix &matrix, AutoResolventOptions options)
    : impl_(std::make_unique<Impl>()) {
    if (matrix.n_rows() <= options.dense_limit) {
        impl_->dense = std::make_unique<DenseResolventSolver>(matrix);
        return;
    }
    if (!sparse_resolvent_available()) {
        throw std::runtime_error("large shifted systems require the sparse complex backend");
    }
    impl_->sparse = std::make_unique<SparseResolventSolver>(
        matrix, SparseResolventOptions{.symmetric_pattern = options.symmetric_pattern});
}

AutoResolventSolver::~AutoResolventSolver() = default;
AutoResolventSolver::AutoResolventSolver(AutoResolventSolver &&) noexcept = default;
AutoResolventSolver &AutoResolventSolver::operator=(AutoResolventSolver &&) noexcept = default;

idx AutoResolventSolver::size() const noexcept {
    return impl_->dense ? impl_->dense->size() : impl_->sparse->size();
}

void AutoResolventSolver::factorize(cplx shift) {
    if (impl_->dense) {
        impl_->dense->factorize(shift);
    } else {
        impl_->sparse->factorize(shift);
    }
}

void AutoResolventSolver::solve(const std::vector<cplx> &rhs, std::vector<cplx> &result) const {
    if (impl_->dense) {
        impl_->dense->solve(rhs, result);
    } else {
        impl_->sparse->solve(rhs, result);
    }
}

std::vector<std::vector<cplx>>
AutoResolventSolver::solve(const std::vector<std::vector<cplx>> &right_hand_sides) const {
    return impl_->dense ? impl_->dense->solve(right_hand_sides)
                        : impl_->sparse->solve(right_hand_sides);
}

} // namespace num
