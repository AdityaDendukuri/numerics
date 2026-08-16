/// @file 06_pde_poisson_solver.cpp
/// @brief 2D/3D Scalar/Vector Fields & Manufactured Poisson PDE Solvers.
#include <numerics.hpp>
#include <iostream>
#include <cmath>

int main() {
    using namespace num;

    idx nx = 16, ny = 16, nz = 16;
    real dx = 1.0 / (nx + 1), dy = 1.0 / (ny + 1), dz = 1.0 / (nz + 1);

    // 1. Scalar Field 3D creation and inspection
    ScalarField3D f(nx, ny, nz, dx, dy, dz);
    for (idx i = 0; i < nx; ++i)
        for (idx j = 0; j < ny; ++j)
            for (idx k = 0; k < nz; ++k)
                f(i, j, k) = std::sin(M_PI * (i + 1) * dx) * std::sin(M_PI * (j + 1) * dy);

    std::cout << "ScalarField3D created. Storage size = " << f.size() << " grid points.\n";

    // 2. Manufactured Solution Poisson PDE Solver: nabla^2 phi = source
    ScalarField3D source(nx, ny, nz, dx, dy, dz);
    for (idx i = 0; i < nx; ++i)
        for (idx j = 0; j < ny; ++j)
            for (idx k = 0; k < nz; ++k)
                source(i, j, k) = -2.0 * M_PI * M_PI * f(i, j, k);

    ScalarField3D phi(nx, ny, nz, dx, dy, dz);
    FieldSolver::solve_poisson(phi, source, 1e-6, 500);

    std::cout << "Poisson PDE Solved. Center point phi(nx/2, ny/2, nz/2) = " << phi(nx/2, ny/2, nz/2) << "\n";

    return 0;
}
