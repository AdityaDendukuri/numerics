/// @file linear/solvers/auto_resolvent.hpp
/// @brief Automatic dense/sparse shifted-resolvent selection.
#pragma once

#include "linear/solvers/dense_resolvent.hpp"
#include <stdexcept>

#include "core/types.hpp"
#include "linear/solvers/sparse_resolvent.hpp"
#include "linear/sparse/sparse.hpp"
#include <memory>
#include <vector>

namespace num {

/// Dense/sparse cutoff and optional sparse symbolic symmetry hint.
struct AutoResolventOptions {
    idx dense_limit = 128;
    bool symmetric_pattern = false;
};

/// Reusable solver for shifted systems (zI-A)x=b with automatic backend choice.
class AutoResolventSolver {
  public:
    /// Store A and select the dense or sparse shifted-system implementation.
    explicit AutoResolventSolver(const SparseMatrix &matrix, AutoResolventOptions options = {});
    ~AutoResolventSolver();
    AutoResolventSolver(AutoResolventSolver &&) noexcept;
    AutoResolventSolver &operator=(AutoResolventSolver &&) noexcept;
    AutoResolventSolver(const AutoResolventSolver &) = delete;
    AutoResolventSolver &operator=(const AutoResolventSolver &) = delete;

    /// Return the order of A.
    [[nodiscard]] idx size() const noexcept;
    /// Factor the shifted matrix zI-A for subsequent solves.
    void factorize(cplx shift);
    /// Solve the currently factored shifted system.
    void solve(const std::vector<cplx> &rhs, std::vector<cplx> &result) const;
    /// Solve several right-hand sides against the current shift.
    [[nodiscard]] std::vector<std::vector<cplx>>
    solve(const std::vector<std::vector<cplx>> &right_hand_sides) const;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

struct AutoResolventSolver::Impl {
    std::unique_ptr<DenseResolventSolver> dense;
    std::unique_ptr<SparseResolventSolver> sparse;
};

inline AutoResolventSolver::AutoResolventSolver(const SparseMatrix &matrix, AutoResolventOptions options)
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

inline AutoResolventSolver::~AutoResolventSolver() = default;
inline AutoResolventSolver::AutoResolventSolver(AutoResolventSolver &&) noexcept = default;
inline AutoResolventSolver &AutoResolventSolver::operator=(AutoResolventSolver &&) noexcept = default;

inline idx AutoResolventSolver::size() const noexcept {
    return impl_->dense ? impl_->dense->size() : impl_->sparse->size();
}

inline void AutoResolventSolver::factorize(cplx shift) {
    if (impl_->dense) {
        impl_->dense->factorize(shift);
    } else {
        impl_->sparse->factorize(shift);
    }
}

inline void AutoResolventSolver::solve(const std::vector<cplx> &rhs, std::vector<cplx> &result) const {
    if (impl_->dense) {
        impl_->dense->solve(rhs, result);
    } else {
        impl_->sparse->solve(rhs, result);
    }
}

inline std::vector<std::vector<cplx>>
AutoResolventSolver::solve(const std::vector<std::vector<cplx>> &right_hand_sides) const {
    return impl_->dense ? impl_->dense->solve(right_hand_sides)
                        : impl_->sparse->solve(right_hand_sides);
}

} // namespace num
