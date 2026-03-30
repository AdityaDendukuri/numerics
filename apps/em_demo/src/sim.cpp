/// @file src/sim.cpp
/// @brief ElectricSolver -- thin wrapper around num::FieldSolver::solve_var_poisson.
#include "em_demo/sim.hpp"
#include <algorithm>

namespace physics {

num::SolverResult ElectricSolver::solve_potential(ScalarField3D& phi,
                                                   const ScalarField3D& sigma,
                                                   const std::vector<ElectrodeBC>& bcs,
                                                   double tol, int max_iter) {
    std::vector<num::FieldSolver::DirichletBC> dbc;
    dbc.reserve(bcs.size());
    for (const auto& e : bcs)
        dbc.push_back({e.flat_idx, static_cast<double>(e.voltage)});
    return num::FieldSolver::solve_var_poisson(phi, sigma, dbc, tol, max_iter);
}

ScalarField3D ElectricSolver::joule_heating(const ScalarField3D& sigma,
                                             const ScalarField3D& phi) {
    const num::Grid3D& g  = phi.grid();
    const num::Grid3D& sg = sigma.grid();
    const int nx = g.nx(), ny = g.ny(), nz = g.nz();
    const double inv2dx = 1.0 / (2.0 * g.dx());

    ScalarField3D Q(nx, ny, nz, phi.dx(), phi.ox(), phi.oy(), phi.oz());
    for (int k = 0; k < nz; ++k)
    for (int j = 0; j < ny; ++j)
    for (int i = 0; i < nx; ++i) {
        int ip = std::min(i+1,nx-1), im = std::max(i-1,0);
        int jp = std::min(j+1,ny-1), jm = std::max(j-1,0);
        int kp = std::min(k+1,nz-1), km = std::max(k-1,0);
        double dpdx = (g(ip,j,k) - g(im,j,k)) * inv2dx;
        double dpdy = (g(i,jp,k) - g(i,jm,k)) * inv2dx;
        double dpdz = (g(i,j,kp) - g(i,j,km)) * inv2dx;
        Q.grid().set(i, j, k, sg(i,j,k) * (dpdx*dpdx + dpdy*dpdy + dpdz*dpdz));
    }
    return Q;
}

} // namespace physics
