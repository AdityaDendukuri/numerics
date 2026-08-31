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
using MatrixFreeLaplacian2D = operators::Laplacian2D;
using MatrixFreeBackwardEuler2D = operators::BackwardEuler2D;

/// @brief Materialize the 5-point discrete Laplacian as an assembled CSR SparseMatrix.
inline SparseMatrix laplacian_sparse_2d(int N) {
    return operators::Laplacian2D(N).to_sparse();
}

/// @brief Materialize the 2D Backward Euler system matrix as an assembled CSR SparseMatrix.
inline SparseMatrix backward_euler_matrix(int N, double coeff) {
    return operators::BackwardEuler2D(N, coeff).to_sparse();
}

inline SparseMatrix backward_euler_matrix(const Grid2D &grid, double coeff) {
    return backward_euler_matrix(grid.N, coeff);
}

/// @brief Pre-assembled SparseMatrix wrapper for 2D Backward Euler diffusion.
class BackwardEulerOperator2D final {
  public:
    using properties = property::spd;
    using domain_type = Vector;
    using codomain_type = Vector;
    using math_propositions = math::type_list<axiom::positive_definite>;

    BackwardEulerOperator2D(int N, double coeff) : A_(validated_matrix(N, coeff)) {}

    void apply(const Vector &x, Vector &y) const { sparse_matvec(A_, x, y); }
    [[nodiscard]] idx rows() const noexcept { return A_.n_rows(); }
    [[nodiscard]] idx cols() const noexcept { return A_.n_cols(); }
    [[nodiscard]] const SparseMatrix &matrix() const noexcept { return A_; }

  private:
    [[nodiscard]] static SparseMatrix validated_matrix(int N, double coeff) {
        if (coeff < 0.0) {
            throw std::invalid_argument(
                "BackwardEulerOperator2D: SPD construction requires nonnegative coefficient");
        }
        return backward_euler_matrix(N, coeff);
    }

    SparseMatrix A_;
};

inline void diffusion_step_2d(Vector &u, int N, double coeff, Backend b = backend::dflt) {
    Vector lap(u.size());
    laplacian_stencil_2d_periodic(u, lap, N);
    axpy(coeff, lap, u, b);
}

inline void diffusion_step_2d_dirichlet(Vector &u, int N, double coeff, Backend b = backend::dflt) {
    Vector lap(u.size());
    laplacian_stencil_2d(u, lap, N);
    axpy(coeff, lap, u, b);
}

inline void diffusion_step_2d_4th_dirichlet(Vector &u, int N, double coeff,
                                            Backend b = backend::dflt) {
    Vector lap(u.size());
    laplacian_stencil_2d_4th(u, lap, N);
    axpy(coeff, lap, u, b);
}

inline void diffusion_step_2d_dirichlet(ScalarField2D &g, double coeff, Backend b = backend::dflt) {
    diffusion_step_2d_dirichlet(g.vec(), g.N(), coeff, b);
}

inline void diffusion_step_2d_4th_dirichlet(ScalarField2D &g, double coeff,
                                            Backend b = backend::dflt) {
    diffusion_step_2d_4th_dirichlet(g.vec(), g.N(), coeff, b);
}

inline BackwardEulerOperator2D backward_euler_operator(int N, double coeff) {
    return BackwardEulerOperator2D(N, coeff);
}

inline BackwardEulerOperator2D backward_euler_operator(const Grid2D &grid, double coeff) {
    return backward_euler_operator(grid.N, coeff);
}

inline LinearSolver make_cg_solver(const SparseMatrix &A, real tol = 1e-6) {
    return [&A, tol](const Vector &rhs, Vector &x) {
        operators::SparseOp op(A);
        return cg(operators::assume_spd(op), rhs, x, tol);
    };
}

} // namespace num::pde

namespace num::math {

template <>
struct model_of<pde::BackwardEulerOperator2D> {
    using laws = type_list<law::linear_map>;
};

} // namespace num::math

namespace num {
using pde::make_cg_solver;
}
