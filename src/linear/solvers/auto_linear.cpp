#include "linear/solvers/auto_linear.hpp"
#include "linear/factorization/lu.hpp"
#include "linear/sparse/klu.hpp"
#include <optional>
#include <stdexcept>

namespace num {

struct auto_linear_solver::Impl {
    idx n = 0;
    std::optional<lu_result<real>> dense_factor;
    std::unique_ptr<klu_factorization> sparse_factor;
};

auto_linear_solver::auto_linear_solver(const spmat &matrix, auto_linear_options options)
    : impl_(std::make_unique<Impl>()) {
    if (matrix.n_rows() != matrix.n_cols()) {
        throw std::invalid_argument("auto_linear_solver requires a square matrix");
    }
    impl_->n = matrix.n_rows();
    if (matrix.n_rows() > options.dense_limit && klu_available()) {
        impl_->sparse_factor = std::make_unique<klu_factorization>(matrix);
    } else {
        // Squareness was rejected above, so the invariant holds here.
        impl_->dense_factor = lu(dense(matrix));
        if (impl_->dense_factor->singular) {
            throw std::runtime_error("auto_linear_solver encountered a singular matrix");
        }
    }
}

auto_linear_solver::~auto_linear_solver() = default;
auto_linear_solver::auto_linear_solver(auto_linear_solver &&) noexcept = default;
auto_linear_solver &auto_linear_solver::operator=(auto_linear_solver &&) noexcept = default;

idx auto_linear_solver::size() const noexcept {
    return impl_ ? impl_->n : 0;
}

void solve(const auto_linear_solver &factor, const vec<real> &rhs, vec<real> &solution) {
    if (factor.impl_->sparse_factor) {
        solve(*factor.impl_->sparse_factor, rhs, solution);
    } else {
        solve(*factor.impl_->dense_factor, rhs, solution);
    }
}

void solve(const auto_linear_solver &factor, const mat<real> &rhs, mat<real> &solution) {
    if (factor.impl_->sparse_factor) {
        solve(*factor.impl_->sparse_factor, rhs, solution);
    } else {
        solve(*factor.impl_->dense_factor, rhs, solution);
    }
}

void solve_transpose(const auto_linear_solver &factor, const vec<real> &rhs, vec<real> &solution) {
    if (factor.impl_->sparse_factor) {
        solve_transpose(*factor.impl_->sparse_factor, rhs, solution);
    } else {
        solve_transpose(*factor.impl_->dense_factor, rhs, solution);
    }
}

void solve_transpose(const auto_linear_solver &factor, const mat<real> &rhs, mat<real> &solution) {
    if (factor.impl_->sparse_factor) {
        solve_transpose(*factor.impl_->sparse_factor, rhs, solution);
    } else {
        solve_transpose(*factor.impl_->dense_factor, rhs, solution);
    }
}

} // namespace num
