/// @file examples/heat_demo.cpp
/// @brief 2D heat equation: implicit backward Euler via sparse CG.
///
/// PDE:  u_t = kappa * nabla^2 u  on [0,1]^2,  Dirichlet BCs.
///
/// Each step solves  A * u^{n+1} = u^n  where A = I - coeff*L  (sparse, SPD).
/// Implicit stepping removes the dt < 0.25 h^2 stability constraint.

#include "numerics.hpp"
#include <string>

using namespace num;

int main() {
  const int N = 64;
  const double kappa = 1.0, sigma = 0.06, T_end = 0.012;
  const double h = 1.0 / (N + 1);
  const double dt = 4.0 * h * h;
  const int nstep = static_cast<int>(T_end / dt) + 1;

  // Build the implicit system: A = I - (kappa*dt/h^2)*L,
  // solve A u^{n+1} = u^n. A is sparse SPD
  // conjugate gradient converges in O(N) iterations
  const double coeff = kappa * dt / (h * h);
  Grid2D grid{N, h};
  SparseMatrix A = pde::backward_euler_matrix(grid, coeff);
  LinearSolver solver = make_cg_solver(A);

  // function defining a gaussian distribution
  auto init_val = [=](double x, double y) {
    return gaussian2d(x, y, 0.5, 0.5, sigma);
  };

  ScalarField2D u0(grid, init_val);
  ScalarField2D u = u0;
  solve(u, BackwardEuler{.solver=solver,
                         .dt=dt,
                         .nstep=nstep});

  // plotting
  plt::subplot(1, 2);
  plt::heatmap(u0);
  plt::title("t = 0");

  plt::next();
  plt::heatmap(u);
  plt::title("t = " + std::to_string(T_end).substr(0, 6));

  plt::savefig("heat_demo.png");
}
