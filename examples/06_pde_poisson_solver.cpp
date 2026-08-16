/// @file 06_pde_poisson_solver.cpp
/// @brief 2D/3D Scalar/Vector Fields & Manufactured Poisson PDE Solvers.
#include <numerics.hpp>
#include <iostream>
#include <cmath>

int main() {
    using namespace num;

    idx nx = 16, ny = 16, nz = 16;
    real dx = 1.0 / (nx + 1), dy = 1.0 / (ny + 1), dz = 1.0 / (nz + 1);

    ScalarField3D f(nx, ny, nz, dx, dy, dz);
    for (idx i = 0; i < nx; ++i)
        for (idx j = 0; j < ny; ++j)
            for (idx k = 0; k < nz; ++k)
                f(i, j, k) = std::sin(M_PI * (i + 1) * dx) * std::sin(M_PI * (j + 1) * dy);

    ScalarField3D source(nx, ny, nz, dx, dy, dz);
    for (idx i = 0; i < nx; ++i)
        for (idx j = 0; j < ny; ++j)
            for (idx k = 0; k < nz; ++k)
                source(i, j, k) = -2.0 * M_PI * M_PI * f(i, j, k);

    ScalarField3D phi(nx, ny, nz, dx, dy, dz);
    FieldSolver::solve_poisson(phi, source, 1e-6, 500);

    // Extract 1D middle slice phi(x, ny/2, nz/2)
    std::vector<double> x_grid, phi_slice;
    for (idx i = 0; i < nx; ++i) {
        x_grid.push_back((i + 1) * dx);
        phi_slice.push_back(phi(i, ny / 2, nz / 2));
    }

    std::cout << "Poisson PDE Solved. Center point phi(nx/2, ny/2, nz/2) = " << phi(nx/2, ny/2, nz/2) << "\n";

    plt::plot(x_grid, phi_slice, "phi(x, y_mid, z_mid)", "lines");
    plt::title("06 PDE Field Solver: 1D Mid-Slice Potential Phi(x)");
    plt::xlabel("Position x");
    plt::ylabel("Potential Phi");
    plt::show_dumb(100, 20);

    return 0;
}
