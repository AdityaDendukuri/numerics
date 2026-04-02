/// @file examples/heat_demo.cpp
/// @brief 2D heat equation: implicit backward Euler via sparse CG.
///
/// PDE:  u_t = kappa * nabla^2 u  on [0,1]^2,  Dirichlet BCs.
///
/// Each step solves  A * u^{n+1} = u^n  where A = I - coeff*L  (sparse, SPD).
/// Implicit stepping removes the dt < 0.25 h^2 stability constraint.

#include "numerics.hpp"
#include <string>

int main() {
  const int N = 64;
  const double kappa = 1.0;
  const double sigma = 0.06;
  const double T_end = 0.012;
  const double h = 1.0 / (N + 1); // N interior nodes; boundaries fixed at 0
  const double dt = 4.0 * h * h;  // 16x larger than explicit limit (0.25 h^2)
  const int nstep = static_cast<int>(T_end / dt) + 1;
  const double coeff = kappa * dt / (h * h); // kappa*dt/h^2

  // Sparse backward Euler system matrix  A = I - coeff*L
  num::ScalarField2D u(N, h);
  num::SparseMatrix A = num::pde::backward_euler_matrix_2d(u, coeff);
  num::SolveStep solver = num::pde::make_cg_solver(A);

  // Initial condition: 2D Gaussian centred at (0.5, 0.5)
  auto initial_temperature = [=](double x, double y) {
    return num::gaussian2d(x, y, 0.5, 0.5, sigma);
  };

  // fill grid with the initial condition
  num::fill_grid(u, initial_temperature);

  // callback function should save snapshiots for plotting
  num::ScalarField2D u0(N, h);
  auto save_snapshots = [&](int step, double /*t*/,
                            const num::ScalarField2D &state) {
    if (step == 0) {
      u0 = state;
    }
  };

  num::pde::diffusion_2d(u, solver, {nstep, dt}, save_snapshots);

  num::plt::subplot(2, 1);
  num::plt::heatmap(u0);
  num::plt::title("t = 0");
  num::plt::xlabel("x");
  num::plt::ylabel("y");
  num::plt::next();

  num::plt::heatmap(u);
  num::plt::title("t = " + std::to_string(T_end).substr(0, 6));
  num::plt::xlabel("x");

  num::plt::savefig("heat_demo.png");
}
