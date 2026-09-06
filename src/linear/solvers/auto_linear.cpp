#include "linear/solvers/auto_linear.hpp"
#include "linear/factorization/lu.hpp"
#include "linear/sparse/klu.hpp"
#include <optional>
#include <stdexcept>

namespace num {

struct auto_linear_solver::Impl {
    idx n = 0;
    std::optional<lu_result> dense_factor;
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
        impl_->dense_factor = lu(assume_square(dense(matrix)));
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

void auto_linear_solver::solve(const vec &rhs, vec &solution) const {
    if (impl_->sparse_factor) {
        impl_->sparse_factor->solve(rhs, solution);
    } else {
        lu_solve(*impl_->dense_factor, rhs, solution);
    }
}

void auto_linear_solver::solve(const mat &rhs, mat &solution) const {
    if (impl_->sparse_factor) {
        impl_->sparse_factor->solve(rhs, solution);
    } else {
        lu_solve(*impl_->dense_factor, rhs, solution);
    }
}

void auto_linear_solver::solve_transpose(const vec &rhs, vec &solution) const {
    if (impl_->sparse_factor) {
        impl_->sparse_factor->solve_transpose(rhs, solution);
    } else {
        lu_solve_transpose(*impl_->dense_factor, rhs, solution);
    }
}

void auto_linear_solver::solve_transpose(const mat &rhs, mat &solution) const {
    if (impl_->sparse_factor) {
        impl_->sparse_factor->solve_transpose(rhs, solution);
    } else {
        lu_solve_transpose(*impl_->dense_factor, rhs, solution);
    }
}

void auto_linear_solver::solve_in_place(vec &right_hand_side) const {
    vec solution(right_hand_side.size(), 0.0);
    solve(right_hand_side, solution);
    right_hand_side = std::move(solution);
}

void auto_linear_solver::solve_in_place(mat &right_hand_sides) const {
    mat solution;
    solve(right_hand_sides, solution);
    right_hand_sides = std::move(solution);
}

} // namespace num
