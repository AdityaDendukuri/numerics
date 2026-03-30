/// @file include/em_demo/sim.hpp
/// @brief EM-specific field types and solvers.
///
/// ScalarField3D, VectorField3D, FieldSolver, and MagneticSolver now live in
/// the numerics library (include/spatial/fields.hpp) in the num:: namespace.
/// This file brings them into the physics:: namespace and adds the
/// EM-specific ElectricSolver (variable-conductivity current flow).
#pragma once

#include "spatial/fields.hpp"
#include "linalg/solvers/solvers.hpp"
#include <vector>

namespace physics {

using ScalarField3D  = num::ScalarField3D;
using VectorField3D  = num::VectorField3D;
using FieldSolver    = num::FieldSolver;
using MagneticSolver = num::MagneticSolver;

// ElectrodeBC + ElectricSolver

/// A grid node with a fixed voltage (Dirichlet BC for current flow).
struct ElectrodeBC {
    int   flat_idx;  ///< Flat grid index: k*ny*nx + j*nx + i
    float voltage;   ///< Applied voltage [V]
};

class ElectricSolver {
public:
    /// Solve div(sigma*grad(phi)) = 0 with Dirichlet BCs at electrode nodes,
    /// Neumann (zero normal flux) on all remaining boundaries.
    ///
    /// Uses symmetric elimination so the system is SPD (CG converges).
    static num::SolverResult solve_potential(ScalarField3D& phi,
                                             const ScalarField3D& sigma,
                                             const std::vector<ElectrodeBC>& bcs,
                                             double tol      = 1e-6,
                                             int    max_iter = 500);

    /// Compute Joule heating power density Q = sigma*|grad(phi)|^2 [W/m^3].
    static ScalarField3D joule_heating(const ScalarField3D& sigma,
                                       const ScalarField3D& phi);
};

} // namespace physics
