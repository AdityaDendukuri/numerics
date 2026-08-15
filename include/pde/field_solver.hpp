/// @file pde/field_solver.hpp
/// @brief Elliptic solvers and vector calculus on 3D field containers.
#pragma once

#include "fields/field3d.hpp"
#include "linalg/solvers/solvers.hpp"
#include <vector>

namespace num {

class FieldSolver {
public:
  /// Dirichlet boundary condition: fix phi = value at grid node flat_idx.
  struct DirichletBC {
    int flat_idx; ///< k*ny*nx + j*nx + i
    double value;
  };

  /// @brief Solve \f$\Delta\phi=s\f$ with zero Dirichlet boundaries.
  static SolverResult solve_poisson(ScalarField3D& phi,
                                    const ScalarField3D& source,
                                    double tol = 1e-6,
                                    int max_iter = 500);

  /// @brief Solve \f$\nabla\cdot(c\nabla\phi)=0\f$ with Dirichlet data.
  static SolverResult solve_var_poisson(ScalarField3D& phi,
                                        const ScalarField3D& coeff,
                                        const std::vector<DirichletBC>& bcs,
                                        double tol = 1e-6,
                                        int max_iter = 500);

  /// @brief Compute \f$\nabla\phi\f$.
  static VectorField3D gradient(const ScalarField3D& phi);

  /// @brief Compute \f$\nabla\cdot f\f$.
  static ScalarField3D divergence(const VectorField3D& f);

  /// @brief Compute \f$\nabla\times A\f$.
  static VectorField3D curl(const VectorField3D& A);
};

class MagneticSolver {
public:
  static constexpr double MU0 = 1.2566370614e-6; ///< mu_0 [H/m]

  /// Compute current density J = -sigma*grad(phi) [A/m^2].
  static VectorField3D current_density(const ScalarField3D& sigma,
                                       const ScalarField3D& phi);

  /// Solve for static magnetic field B given current density J.
  /// Solves Laplacian(A) = -mu0*J (Coulomb gauge, Dirichlet A=0) via three CG
  /// solves, then returns B = curl(A).
  static VectorField3D solve_magnetic_field(const VectorField3D& J,
                                            double tol = 1e-6,
                                            int max_iter = 500);
};

} // namespace num
