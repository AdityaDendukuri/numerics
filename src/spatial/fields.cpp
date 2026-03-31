/// @file src/spatial/fields.cpp
/// @brief Implementations for num::ScalarField3D, VectorField3D, FieldSolver,
/// MagneticSolver.
#include "spatial/fields.hpp"
#include "linalg/solvers/solvers.hpp"
#include "spatial/stencil.hpp"
#include <algorithm>
#include <cmath>

namespace num {

// ============================================================
// ScalarField3D
// ============================================================

ScalarField3D::ScalarField3D(int   nx,
                             int   ny,
                             int   nz,
                             float dx,
                             float ox,
                             float oy,
                             float oz)
    : grid_(nx, ny, nz, static_cast<double>(dx))
    , ox_(ox)
    , oy_(oy)
    , oz_(oz) {}

float ScalarField3D::sample(float x, float y, float z) const {
    const float gx = (x - ox_) / dx();
    const float gy = (y - oy_) / dx();
    const float gz = (z - oz_) / dx();

    if (gx < 0 || gx >= nx() - 1 || gy < 0 || gy >= ny() - 1 || gz < 0
        || gz >= nz() - 1)
        return 0.0f;

    const int   i0 = static_cast<int>(gx);
    const int   j0 = static_cast<int>(gy);
    const int   k0 = static_cast<int>(gz);
    const float tx = gx - i0, ty = gy - j0, tz = gz - k0;

    auto v = [&](int di, int dj, int dk) {
        return static_cast<float>(grid_(i0 + di, j0 + dj, k0 + dk));
    };
    return (1 - tz)
               * ((1 - ty) * ((1 - tx) * v(0, 0, 0) + tx * v(1, 0, 0))
                  + ty * ((1 - tx) * v(0, 1, 0) + tx * v(1, 1, 0)))
           + tz
                 * ((1 - ty) * ((1 - tx) * v(0, 0, 1) + tx * v(1, 0, 1))
                    + ty * ((1 - tx) * v(0, 1, 1) + tx * v(1, 1, 1)));
}

// ============================================================
// VectorField3D
// ============================================================

VectorField3D::VectorField3D(int   nx,
                             int   ny,
                             int   nz,
                             float dx,
                             float ox,
                             float oy,
                             float oz)
    : x(nx, ny, nz, dx, ox, oy, oz)
    , y(nx, ny, nz, dx, ox, oy, oz)
    , z(nx, ny, nz, dx, ox, oy, oz) {}

std::array<float, 3> VectorField3D::sample(float px, float py, float pz) const {
    return {x.sample(px, py, pz), y.sample(px, py, pz), z.sample(px, py, pz)};
}

void VectorField3D::scale(float s) {
    auto sc = [&](Grid3D& g) {
        auto v = g.to_vector();
        num::scale(v, static_cast<real>(s));
        g.from_vector(v);
    };
    sc(x.grid());
    sc(y.grid());
    sc(z.grid());
}

// ============================================================
// FieldSolver
// ============================================================

SolverResult FieldSolver::solve_poisson(ScalarField3D&       phi,
                                        const ScalarField3D& source,
                                        double               tol,
                                        int                  max_iter) {
    const Grid3D& g  = phi.grid();
    const int     nx = g.nx(), ny = g.ny(), nz = g.nz();
    const double  inv_dx2 = 1.0 / (g.dx() * g.dx());
    const int     N       = nx * ny * nz;

    auto flat = [&](int i, int j, int k) -> idx {
        return static_cast<idx>(k * ny * nx + j * nx + i);
    };

    Vector b(N, 0.0);
    for (int k = 1; k < nz - 1; ++k) {
        for (int j = 1; j < ny - 1; ++j) {
            for (int i = 1; i < nx - 1; ++i) {
                b[flat(i, j, k)] = -source.grid()(i, j, k);
            }
        }
    }

    Vector xv = phi.grid().to_vector();

    auto matvec = [&](const Vector& v, Vector& Av) {
        neg_laplacian_3d(v, Av, nx, ny, nz, inv_dx2);
    };
    auto result = cg_matfree(matvec, b, xv, tol, static_cast<idx>(max_iter));
    phi.grid().from_vector(xv);
    return result;
}

VectorField3D FieldSolver::gradient(const ScalarField3D& phi) {
    VectorField3D out(phi.nx(),
                      phi.ny(),
                      phi.nz(),
                      phi.dx(),
                      phi.ox(),
                      phi.oy(),
                      phi.oz());
    gradient_3d(phi.grid(), out.x.grid(), out.y.grid(), out.z.grid());
    return out;
}

ScalarField3D FieldSolver::divergence(const VectorField3D& f) {
    ScalarField3D out(f.x.nx(),
                      f.x.ny(),
                      f.x.nz(),
                      f.x.dx(),
                      f.x.ox(),
                      f.x.oy(),
                      f.x.oz());
    divergence_3d(f.x.grid(), f.y.grid(), f.z.grid(), out.grid());
    return out;
}

VectorField3D FieldSolver::curl(const VectorField3D& A) {
    VectorField3D
        B(A.x.nx(), A.x.ny(), A.x.nz(), A.x.dx(), A.x.ox(), A.x.oy(), A.x.oz());
    curl_3d(A.x.grid(),
            A.y.grid(),
            A.z.grid(),
            B.x.grid(),
            B.y.grid(),
            B.z.grid());
    return B;
}

// ============================================================
// MagneticSolver
// ============================================================

VectorField3D MagneticSolver::current_density(const ScalarField3D& sigma,
                                              const ScalarField3D& phi) {
    VectorField3D J  = FieldSolver::gradient(phi);
    const Grid3D& sg = sigma.grid();
    const int     nx = sg.nx(), ny = sg.ny(), nz = sg.nz();
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                const double neg_s = -sg(i, j, k);
                J.x.grid()(i, j, k) *= neg_s;
                J.y.grid()(i, j, k) *= neg_s;
                J.z.grid()(i, j, k) *= neg_s;
            }
    return J;
}

VectorField3D MagneticSolver::solve_magnetic_field(const VectorField3D& J,
                                                   double               tol,
                                                   int max_iter) {
    const int   nx = J.x.nx(), ny = J.x.ny(), nz = J.x.nz();
    const float dx = J.x.dx(), ox = J.x.ox(), oy = J.x.oy(), oz = J.x.oz();

    auto make_source = [&](const ScalarField3D& Jc) {
        ScalarField3D src(nx, ny, nz, dx, ox, oy, oz);
        const Grid3D& gJ = Jc.grid();
        Grid3D&       gs = src.grid();
        for (int k = 0; k < nz; ++k)
            for (int j = 0; j < ny; ++j)
                for (int i = 0; i < nx; ++i)
                    gs(i, j, k) = -MU0 * gJ(i, j, k);
        return src;
    };

    ScalarField3D Ax(nx, ny, nz, dx, ox, oy, oz);
    ScalarField3D Ay(nx, ny, nz, dx, ox, oy, oz);
    ScalarField3D Az(nx, ny, nz, dx, ox, oy, oz);

    FieldSolver::solve_poisson(Ax, make_source(J.x), tol, max_iter);
    FieldSolver::solve_poisson(Ay, make_source(J.y), tol, max_iter);
    FieldSolver::solve_poisson(Az, make_source(J.z), tol, max_iter);

    VectorField3D A(nx, ny, nz, dx, ox, oy, oz);
    A.x = Ax;
    A.y = Ay;
    A.z = Az;
    return FieldSolver::curl(A);
}

} // namespace num
