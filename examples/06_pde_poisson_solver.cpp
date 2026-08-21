/// @file 06_pde_poisson_solver.cpp
/// @brief 2D/3D Scalar/Vector Fields & Manufactured Poisson PDE Solvers.
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numerics.hpp>

int main() {
  using namespace num;

  const idx nx = 16, ny = 16, nz = 16;
  const real dx = 1.0 / (nx - 1), dy = 1.0 / (ny - 1), dz = 1.0 / (nz - 1);

  // This mode vanishes on every boundary face.
  ScalarField3D exact(nx, ny, nz, dx, dy, dz);
  for (idx i = 0; i < nx; ++i) {
    for (idx j = 0; j < ny; ++j) {
      for (idx k = 0; k < nz; ++k) {
        exact(i, j, k) =
          std::sin(M_PI * i * dx) * std::sin(M_PI * j * dy) * std::sin(M_PI * k * dz);
      }
    }
  }

  // Apply the discrete Laplacian to manufacture a grid-exact source.
  ScalarField3D source(nx, ny, nz, dx, dy, dz);
  for (idx i = 1; i < nx - 1; ++i) {
    for (idx j = 1; j < ny - 1; ++j) {
      for (idx k = 1; k < nz - 1; ++k) {
        source(i, j, k) = (exact(i + 1, j, k) + exact(i - 1, j, k) + exact(i, j + 1, k)
                           + exact(i, j - 1, k) + exact(i, j, k + 1) + exact(i, j, k - 1)
                           - 6.0 * exact(i, j, k))
                          / (dx * dx);
      }
    }
  }

  ScalarField3D phi(nx, ny, nz, dx, dy, dz);
  const SolverResult result = FieldSolver::solve_poisson(phi, source, 1e-10, 500);

  real max_error = 0.0;
  for (idx state = 0; state < phi.size(); ++state) {
    max_error = std::max(max_error, std::abs(phi.vec()[state] - exact.vec()[state]));
  }

  // Extract 1D middle slice phi(x, ny/2, nz/2)
  std::vector<double> x_grid, phi_slice;
  for (idx i = 0; i < nx; ++i) {
    x_grid.push_back((i + 1) * dx);
    phi_slice.push_back(phi(i, ny / 2, nz / 2));
  }

  std::cout << "Poisson converged: " << std::boolalpha << result.converged
            << ", max error = " << max_error << '\n';

  plt::plot(x_grid, phi_slice, "phi(x, y_mid, z_mid)", "lines");
  plt::title("06 PDE Field Solver: 1D Mid-Slice Potential Phi(x)");
  plt::xlabel("Position x");
  plt::ylabel("Potential Phi");
  plt::show_dumb(140, 35);

  return result.converged && max_error < 1e-6 ? 0 : 1;
}
