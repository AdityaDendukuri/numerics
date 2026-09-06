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
struct auto_resolvent_options {
    idx dense_limit = 128;
    bool symmetric_pattern = false;
};

/// Reusable solver for shifted systems (zI-A)x=b with automatic backend choice.
class auto_resolvent_solver {
  public:
    /// Store A and select the dense or sparse shifted-system implementation.
    explicit auto_resolvent_solver(const spmat &matrix, auto_resolvent_options options = {});
    ~auto_resolvent_solver();
    auto_resolvent_solver(auto_resolvent_solver &&) noexcept;
    auto_resolvent_solver &operator=(auto_resolvent_solver &&) noexcept;
    auto_resolvent_solver(const auto_resolvent_solver &) = delete;
    auto_resolvent_solver &operator=(const auto_resolvent_solver &) = delete;

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

struct auto_resolvent_solver::Impl {
    std::unique_ptr<dense_resolvent_solver> dense;
    std::unique_ptr<sparse_resolvent_solver> sparse;
};

inline auto_resolvent_solver::auto_resolvent_solver(const spmat &matrix, auto_resolvent_options options)
    : impl_(std::make_unique<Impl>()) {
    if (matrix.n_rows() <= options.dense_limit) {
        impl_->dense = std::make_unique<dense_resolvent_solver>(matrix);
        return;
    }
    if (!sparse_resolvent_available()) {
        throw std::runtime_error("large shifted systems require the sparse complex backend");
    }
    impl_->sparse = std::make_unique<sparse_resolvent_solver>(
        matrix, sparse_resolvent_options{.symmetric_pattern = options.symmetric_pattern});
}

inline auto_resolvent_solver::~auto_resolvent_solver() = default;
inline auto_resolvent_solver::auto_resolvent_solver(auto_resolvent_solver &&) noexcept = default;
inline auto_resolvent_solver &auto_resolvent_solver::operator=(auto_resolvent_solver &&) noexcept = default;

inline idx auto_resolvent_solver::size() const noexcept {
    return impl_->dense ? impl_->dense->size() : impl_->sparse->size();
}

inline void auto_resolvent_solver::factorize(cplx shift) {
    if (impl_->dense) {
        impl_->dense->factorize(shift);
    } else {
        impl_->sparse->factorize(shift);
    }
}

inline void auto_resolvent_solver::solve(const std::vector<cplx> &rhs, std::vector<cplx> &result) const {
    if (impl_->dense) {
        impl_->dense->solve(rhs, result);
    } else {
        impl_->sparse->solve(rhs, result);
    }
}

inline std::vector<std::vector<cplx>>
auto_resolvent_solver::solve(const std::vector<std::vector<cplx>> &right_hand_sides) const {
    return impl_->dense ? impl_->dense->solve(right_hand_sides)
                        : impl_->sparse->solve(right_hand_sides);
}

} // namespace num
