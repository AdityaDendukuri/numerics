/// @file pde/diffusion.hpp
/// @brief 2D diffusion operators and implicit system builders.
#pragma once

#include "operator/properties.hpp"

#include "container/vector_ops.hpp"

#include "container/vector.hpp"
#include "core/policy.hpp"
#include "fields/grid2d.hpp"
#include "linear/solvers/cg.hpp"
#include "linear/solvers/linear_solver.hpp"
#include "linear/sparse/sparse.hpp"
#include "linear/sparse/sparse_op.hpp"
#include "pde/grid_operators.hpp"
#include "pde/stencil.hpp"
#include <stdexcept>

namespace num::pde {

/// @brief Matrix-free 2D Laplacian.
using matrix_free_laplacian_2d = operators::laplacian_2d;
using matrix_free_backward_euler_2d = operators::backward_euler_2d;

/// @brief 5-point Laplacian as a sparse matrix: \f$\Delta_h\f$.
inline spmat laplacian_sparse_2d(int N) {
    return operators::laplacian_2d(N).to_sparse();
}

/// @brief Backward Euler matrix as a sparse matrix: \f$I - c\,\Delta_h\f$.
inline spmat backward_euler_matrix(int N, double coeff) {
    return operators::backward_euler_2d(N, coeff).to_sparse();
}

inline spmat backward_euler_matrix(const grid2d &grid, double coeff) {
    return backward_euler_matrix(grid.N, coeff);
}

/// @brief Backward Euler operator, preassembled: \f$I - c\,\Delta_h\f$.
class backward_euler_operator_2d final {
  public:
    using domain_type = vec<real>;
    using codomain_type = vec<real>;
    using laws = law::list<law::spd>;

    backward_euler_operator_2d(int N, double coeff) : A_(validated_matrix(N, coeff)) {}

    void apply(const vec<real> &x, vec<real> &y) const { sparse_matvec(A_, x, y); }
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

/// @brief Explicit diffusion step, periodic: \f$u \leftarrow u + c\,\Delta_h u\f$.
inline void diffusion_step_2d(vec<real> &u, int N, double coeff) {
    vec<real> lap(u.size());
    laplacian_stencil_2d_periodic(u, lap, N);
    axpy(coeff, lap, u);
}

/// @brief Explicit diffusion step, zero Dirichlet: \f$u \leftarrow u + c\,\Delta_h u\f$.
inline void diffusion_step_2d_dirichlet(vec<real> &u, int N, double coeff) {
    vec<real> lap(u.size());
    laplacian_stencil_2d(u, lap, N);
    axpy(coeff, lap, u);
}

/// @brief Explicit diffusion step, fourth order: \f$u \leftarrow u + c\,\Delta_h u\f$.
/// Zero Dirichlet boundaries.
inline void diffusion_step_2d_4th_dirichlet(vec<real> &u, int N, double coeff) {
    vec<real> lap(u.size());
    laplacian_stencil_2d_4th(u, lap, N);
    axpy(coeff, lap, u);
}

inline void diffusion_step_2d_dirichlet(scalar_field_2d &g, double coeff) {
    diffusion_step_2d_dirichlet(g.as_vec(), g.N(), coeff);
}

inline void diffusion_step_2d_4th_dirichlet(scalar_field_2d &g, double coeff) {
    diffusion_step_2d_4th_dirichlet(g.as_vec(), g.N(), coeff);
}

/// @brief Backward Euler operator on an \f$N \times N\f$ grid: \f$I - c\,\Delta_h\f$.
inline backward_euler_operator_2d backward_euler_operator(int N, double coeff) {
    return backward_euler_operator_2d(N, coeff);
}

inline backward_euler_operator_2d backward_euler_operator(const grid2d &grid, double coeff) {
    return backward_euler_operator(grid.N, coeff);
}

/// @brief CG linear solver for `A`, held by reference.
inline linear_solver make_cg_solver(const spmat &A, real tol = 1e-6) {
    return [&A, tol](const vec<real> &rhs, vec<real> &x) {
        operators::sparse_op op(A);
        return cg(num::assume_spd(op), rhs, x, {.tolerance = tol});
    };
}

} // namespace num::pde


namespace num {
using pde::make_cg_solver;
}
