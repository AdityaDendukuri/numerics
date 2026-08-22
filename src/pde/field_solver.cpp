/// @file src/pde/field_solver.cpp
/// @brief Implementations for FieldSolver and MagneticSolver.
#include "pde/field_solver.hpp"
#include "linalg/solvers/solvers.hpp"
#include "operator/operator.hpp"

#include "pde/stencil.hpp"
#include <algorithm>
#include <unordered_map>

namespace num {

SolverResult FieldSolver::solve_poisson(ScalarField3D &phi, const ScalarField3D &source, double tol,
                                        int max_iter) {
    const int nx = phi.nx(), ny = phi.ny(), nz = phi.nz();
    const double inv_dx2 = 1.0 / (phi.dx() * phi.dx());
    const idx N = phi.size();

    // RHS = -source on the interior; Dirichlet boundary rows stay 0.
    Vector b(N, 0.0);
    for (int k = 1; k < nz - 1; ++k) {
        for (int j = 1; j < ny - 1; ++j) {
            for (int i = 1; i < nx - 1; ++i) {
                b[phi.grid().flat(i, j, k)] = -source(i, j, k);
            }
        }
    }

    // A = -Laplacian (SPD). phi's own storage is the solution vector, so CG
    // solves in place -- no copy in or out.
    auto A = operators::assume_spd(operators::make_op(
        [&](const Vector &v, Vector &Av) { neg_laplacian_3d(v, Av, nx, ny, nz, inv_dx2); }, N));
    return num::cg(A, b, phi.vec(), tol, static_cast<idx>(max_iter));
}

SolverResult FieldSolver::solve_var_poisson(ScalarField3D &phi, const ScalarField3D &coeff,
                                            const std::vector<DirichletBC> &bcs, double tol,
                                            int max_iter) {
    const int nx = phi.nx(), ny = phi.ny(), nz = phi.nz();
    const idx N = phi.size();
    const double inv_dx2 = 1.0 / (phi.dx() * phi.dx());

    auto flat = [&](int i, int j, int k) -> idx {
        return static_cast<idx>((k * ny * nx) + (j * nx) + i);
    };

    std::unordered_map<int, double> bc_map;
    bc_map.reserve(bcs.size());
    for (const auto &e : bcs) {
        bc_map[e.flat_idx] = e.value;
    }

    constexpr int DI[6] = {1, -1, 0, 0, 0, 0};
    constexpr int DJ[6] = {0, 0, 1, -1, 0, 0};
    constexpr int DK[6] = {0, 0, 0, 0, 1, -1};
    constexpr double penalty = 1e10;

    // Symmetric penalty elimination: fold each Dirichlet value into the RHS of
    // its free neighbours so the operator stays SPD.
    Vector b(N, 0.0);
    for (const auto &e : bcs) {
        b[e.flat_idx] = penalty * e.value;
        const int ei = e.flat_idx % nx;
        const int ej = (e.flat_idx / nx) % ny;
        const int ek = e.flat_idx / (nx * ny);
        for (int d = 0; d < 6; ++d) {
            int ni = ei + DI[d], nj = ej + DJ[d], nk = ek + DK[d];
            if (ni < 0 || ni >= nx || nj < 0 || nj >= ny || nk < 0 || nk >= nz) {
                continue;
            }
            int nidx = flat(ni, nj, nk);
            if (bc_map.count(nidx)) {
                continue;
            }
            double sigma_face = 0.5 * (coeff(ei, ej, ek) + coeff(ni, nj, nk));
            b[nidx] += sigma_face * inv_dx2 * e.value;
        }
    }

    auto matvec = [&](const Vector &v, Vector &Av) {
        for (int k = 0; k < nz; ++k) {
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    int id = flat(i, j, k);
                    if (bc_map.count(id)) {
                        Av[id] = penalty * v[id];
                        continue;
                    }
                    double Av_ijk = 0.0;
                    for (int d = 0; d < 6; ++d) {
                        int ni = std::max(0, std::min(i + DI[d], nx - 1));
                        int nj = std::max(0, std::min(j + DJ[d], ny - 1));
                        int nk = std::max(0, std::min(k + DK[d], nz - 1));
                        int nidx = flat(ni, nj, nk);
                        double c_face = 0.5 * (coeff(i, j, k) + coeff(ni, nj, nk));
                        double v_nb = bc_map.count(nidx) ? 0.0 : v[nidx];
                        Av_ijk += c_face * inv_dx2 * (v[id] - v_nb);
                    }
                    Av[id] = Av_ijk;
                }
            }
        }
    };

    // Solve A phi = b in place; phi's storage is the solution vector.
    auto A = operators::assume_spd(operators::make_op(matvec, N));
    return num::cg(A, b, phi.vec(), tol, static_cast<idx>(max_iter));
}

VectorField3D FieldSolver::gradient(const ScalarField3D &phi) {
    VectorField3D out(phi.nx(), phi.ny(), phi.nz(), phi.dx(), phi.ox(), phi.oy(), phi.oz());
    gradient_3d(phi, out.x, out.y, out.z);
    return out;
}

ScalarField3D FieldSolver::divergence(const VectorField3D &f) {
    ScalarField3D out(f.x.nx(), f.x.ny(), f.x.nz(), f.x.dx(), f.x.ox(), f.x.oy(), f.x.oz());
    divergence_3d(f.x, f.y, f.z, out);
    return out;
}

VectorField3D FieldSolver::curl(const VectorField3D &A) {
    VectorField3D B(A.x.nx(), A.x.ny(), A.x.nz(), A.x.dx(), A.x.ox(), A.x.oy(), A.x.oz());
    curl_3d(A.x, A.y, A.z, B.x, B.y, B.z);
    return B;
}

// MagneticSolver

VectorField3D MagneticSolver::current_density(const ScalarField3D &sigma,
                                              const ScalarField3D &phi) {
    VectorField3D J = FieldSolver::gradient(phi);
    const int nx = sigma.nx(), ny = sigma.ny(), nz = sigma.nz();
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                const double neg_s = -sigma(i, j, k);
                J.x(i, j, k) *= neg_s;
                J.y(i, j, k) *= neg_s;
                J.z(i, j, k) *= neg_s;
            }
        }
    }
    return J;
}

VectorField3D MagneticSolver::solve_magnetic_field(const VectorField3D &J, double tol,
                                                   int max_iter) {
    const int nx = J.x.nx(), ny = J.x.ny(), nz = J.x.nz();
    const float dx = J.x.dx(), ox = J.x.ox(), oy = J.x.oy(), oz = J.x.oz();

    auto make_source = [&](const ScalarField3D &Jc) {
        ScalarField3D src(nx, ny, nz, dx, ox, oy, oz);
        for (int k = 0; k < nz; ++k) {
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    src(i, j, k) = -MU0 * Jc(i, j, k);
                }
            }
        }
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
