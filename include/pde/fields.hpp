/// @file pde/fields.hpp
/// @brief 3D scalar and vector fields on Cartesian grids, with PDE field
/// solvers.
///
/// ScalarField3D  -- potential phi(x,y,z) on a uniform grid with trilinear
/// sampling. VectorField3D  -- three ScalarField3D components; sample() returns
/// trilinear (fx,fy,fz). FieldSolver    -- static PDE utilities (Poisson solve,
/// gradient, divergence, curl). MagneticSolver -- static utilities for J =
/// -sigma*grad(phi) and B = curl(A) via vector Poisson.
#pragma once

#include "spatial/grid3d.hpp"
#include "linalg/solvers/solvers.hpp"
#include <array>
#include <vector>

namespace num {

// ScalarField3D

class ScalarField3D {
  public:
    /// @param nx,ny,nz  Grid resolution
    /// @param dx        Cell size [m]
    /// @param ox,oy,oz  World-space origin
    ScalarField3D(int   nx,
                  int   ny,
                  int   nz,
                  float dx,
                  float ox = 0.0f,
                  float oy = 0.0f,
                  float oz = 0.0f);

    Grid3D& grid() {
        return grid_;
    }
    const Grid3D& grid() const {
        return grid_;
    }

    int nx() const {
        return grid_.nx();
    }
    int ny() const {
        return grid_.ny();
    }
    int nz() const {
        return grid_.nz();
    }
    float dx() const {
        return static_cast<float>(grid_.dx());
    }
    float ox() const {
        return ox_;
    }
    float oy() const {
        return oy_;
    }
    float oz() const {
        return oz_;
    }

    void set(int i, int j, int k, double v) {
        grid_.set(i, j, k, v);
    }
    void fill(double v) {
        grid_.fill(v);
    }

    /// Trilinear interpolation at world position (x,y,z).
    /// Returns 0 outside the grid domain.
    float sample(float x, float y, float z) const;

  private:
    Grid3D grid_;
    float  ox_, oy_, oz_;
};

// VectorField3D

struct VectorField3D {
    ScalarField3D x, y, z; ///< x, y, z components on the same grid layout

    VectorField3D(int   nx,
                  int   ny,
                  int   nz,
                  float dx,
                  float ox = 0.0f,
                  float oy = 0.0f,
                  float oz = 0.0f);

    /// Trilinear-interpolated field vector at world position.
    std::array<float, 3> sample(float px, float py, float pz) const;

    /// Multiply all components by scalar s.
    void scale(float s);
};

// FieldSolver

class FieldSolver {
  public:
    /// Dirichlet boundary condition: fix phi = value at grid node flat_idx.
    struct DirichletBC {
        int    flat_idx; ///< k*ny*nx + j*nx + i
        double value;
    };

    /// Solve Laplacian(phi) = source with phi=0 on all boundaries (Dirichlet).
    /// phi is both the initial guess and the output solution.
    /// Internally solves the SPD system (-Laplacian)phi = -source via
    /// matrix-free CG.
    static SolverResult solve_poisson(ScalarField3D&       phi,
                                      const ScalarField3D& source,
                                      double               tol      = 1e-6,
                                      int                  max_iter = 500);

    /// Solve div(coeff * grad(phi)) = 0 with arbitrary Dirichlet BCs.
    ///
    /// Typical use: current flow in a heterogeneous conductor (coeff =
    /// conductivity sigma). Imposes BCs via symmetric penalty elimination so
    /// the system remains SPD -> CG converges. Neumann (zero normal flux) on
    /// all non-BC boundaries.
    static SolverResult solve_var_poisson(ScalarField3D&                  phi,
                                          const ScalarField3D&            coeff,
                                          const std::vector<DirichletBC>& bcs,
                                          double tol      = 1e-6,
                                          int    max_iter = 500);

    /// Compute grad(phi) via central finite differences (one-sided at
    /// boundaries).
    static VectorField3D gradient(const ScalarField3D& phi);

    /// Compute div(f) via central finite differences.
    static ScalarField3D divergence(const VectorField3D& f);

    /// Compute curl(A) via central finite differences (one-sided at
    /// boundaries).
    static VectorField3D curl(const VectorField3D& A);
};

// MagneticSolver

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
                                              double               tol = 1e-6,
                                              int max_iter             = 500);
};

} // namespace num
