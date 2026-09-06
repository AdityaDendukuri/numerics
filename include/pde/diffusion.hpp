/// @file pde/diffusion.hpp
/// @brief Diffusion operators and implicit system builders for 2D grids.
#pragma once

#include "algebra/properties.hpp"

#include "container/vector_ops.hpp"

#include "container/vector.hpp"
#include "core/policy.hpp"
#include "fields/grid2d.hpp"
#include "linear/solvers/cg.hpp"
#include "linear/solvers/linear_solver.hpp"
#include "linear/sparse/sparse.hpp"
#include "linear/sparse/sparse_op.hpp"
#include "operator/properties.hpp"
#include "pde/grid_operators.hpp"
#include "pde/stencil.hpp"
#include <stdexcept>

namespace num::pde {

/// @brief Aliases to operator module grid/stencil definitions.
using matrix_free_laplacian_2d = operators::laplacian_2d;
using matrix_free_backward_euler_2d = operators::backward_euler_2d;

/// @brief Materialize the 5-point discrete Laplacian as an assembled CSR spmat.
inline spmat laplacian_sparse_2d(int N) {
    return operators::laplacian_2d(N).to_sparse();
}

/// @brief Materialize the 2D Backward Euler system matrix as an assembled CSR spmat.
inline spmat backward_euler_matrix(int N, double coeff) {
    return operators::backward_euler_2d(N, coeff).to_sparse();
}

inline spmat backward_euler_matrix(const grid2d &grid, double coeff) {
    return backward_euler_matrix(grid.N, coeff);
}

/// @brief Pre-assembled spmat wrapper for 2D Backward Euler diffusion.
class backward_euler_operator_2d final {
  public:
    using domain_type = vec;
    using codomain_type = vec;
    using math_laws = math::type_list<law::spd>;

    backward_euler_operator_2d(int N, double coeff) : A_(validated_matrix(N, coeff)) {}

    void apply(const vec &x, vec &y) const { sparse_matvec(A_, x, y); }
    [[nodiscard]] idx rows() const noexcept { return A_.n_rows(); }
    [[nodiscard]] idx cols() const noexcept { return A_.n_cols(); }
    [[nodiscard]] const spmat &matrix() const noexcept { return A_; }

  private:
    [[nodiscard]] static spmat validated_matrix(int N, double coeff) {
        if (coeff < 0.0) {
            throw std::invalid_argument(
                "backward_euler_operator_2d: SPD construction requires nonnegative coefficient");
        }
        return backward_euler_matrix(N, coeff);
    }

    spmat A_;
};

inline void diffusion_step_2d(vec &u, int N, double coeff) {
    vec lap(u.size());
    laplacian_stencil_2d_periodic(u, lap, N);
    axpy(coeff, lap, u);
}

inline void diffusion_step_2d_dirichlet(vec &u, int N, double coeff) {
    vec lap(u.size());
    laplacian_stencil_2d(u, lap, N);
    axpy(coeff, lap, u);
}

inline void diffusion_step_2d_4th_dirichlet(vec &u, int N, double coeff) {
    vec lap(u.size());
    laplacian_stencil_2d_4th(u, lap, N);
    axpy(coeff, lap, u);
}

inline void diffusion_step_2d_dirichlet(scalar_field_2d &g, double coeff) {
    diffusion_step_2d_dirichlet(g.as_vec(), g.N(), coeff);
}

inline void diffusion_step_2d_4th_dirichlet(scalar_field_2d &g, double coeff) {
    diffusion_step_2d_4th_dirichlet(g.as_vec(), g.N(), coeff);
}

inline backward_euler_operator_2d backward_euler_operator(int N, double coeff) {
    return backward_euler_operator_2d(N, coeff);
}

inline backward_euler_operator_2d backward_euler_operator(const grid2d &grid, double coeff) {
    return backward_euler_operator(grid.N, coeff);
}

inline linear_solver make_cg_solver(const spmat &A, real tol = 1e-6) {
    return [&A, tol](const vec &rhs, vec &x) {
        operators::sparse_op op(A);
        return cg(operators::assume_spd(op), rhs, x, tol);
    };
}

} // namespace num::pde

namespace num::math {

template <>
struct claims_of<pde::backward_euler_operator_2d> {
    using type = type_list<law::linear_map>;
};

} // namespace num::math

namespace num {
using pde::make_cg_solver;
}
