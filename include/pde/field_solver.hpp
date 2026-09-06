/// @file pde/field_solver.hpp
/// @brief Elliptic solvers and vector calculus on 3D field containers.
#pragma once

#include "core/types.hpp"
#include "operator/operator.hpp"
#include "pde/stencil.hpp"
#include <algorithm>
#include <unordered_map>

#include "fields/field3d.hpp"
#include "linear/solvers/solvers.hpp"
#include <vector>

namespace num {

class field_solver {
  public:
    /// Dirichlet boundary condition: fix phi = value at grid node flat_idx.
    struct dirichlet_bc {
        int flat_idx; ///< k*ny*nx + j*nx + i
        double value;
    };

    /// @brief Solve \f$\Delta\phi=s\f$ with zero Dirichlet boundaries.
    /// @return `solver_result`: `.iterations`, `.residual` (final residual norm), `.converged`.
    static solver_result solve_poisson(scalar_field_3d &phi, const scalar_field_3d &source,
                                      double tol = 1e-6, int max_iter = 500);

    /// @brief Solve \f$\nabla\cdot(c\nabla\phi)=0\f$ with Dirichlet data.
    /// @return `solver_result`: `.iterations`, `.residual` (final residual norm), `.converged`.
    static solver_result solve_var_poisson(scalar_field_3d &phi, const scalar_field_3d &coeff,
                                          const array<dirichlet_bc> &bcs, double tol = 1e-6,
                                          int max_iter = 500);

    /// @brief Compute \f$\nabla\phi\f$.
    static vector_field_3d gradient(const scalar_field_3d &phi);

    /// @brief Compute \f$\nabla\cdot f\f$.
    static scalar_field_3d divergence(const vector_field_3d &f);

    /// @brief Compute \f$\nabla\times A\f$.
    static vector_field_3d curl(const vector_field_3d &A);
};

class magnetic_solver {
  public:
    static constexpr double MU0 = 1.2566370614e-6; ///< mu_0 [H/m]

    /// Compute current density J = -sigma*grad(phi) [A/m^2].
    static vector_field_3d current_density(const scalar_field_3d &sigma, const scalar_field_3d &phi);

    /// Solve for static magnetic field B given current density J.
    /// Solves Laplacian(A) = -mu0*J (Coulomb gauge, Dirichlet A=0) via three CG
    /// solves, then returns B = curl(A).
    static vector_field_3d solve_magnetic_field(const vector_field_3d &J, double tol = 1e-6,
                                              int max_iter = 500);
};

inline solver_result field_solver::solve_poisson(scalar_field_3d &phi, const scalar_field_3d &source, double tol,
                                        int max_iter) {
    const int nx = phi.nx(), ny = phi.ny(), nz = phi.nz();
    const double inv_dx2 = 1.0 / (phi.dx() * phi.dx());
    const idx N = phi.size();

    // RHS = -source on the interior; Dirichlet boundary rows stay 0.
    vec b(N, 0.0);
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
        [&](const vec &v, vec &Av) { neg_laplacian_3d(v, Av, nx, ny, nz, inv_dx2); }, N));
    return num::cg(A, b, phi.as_vec(), tol, static_cast<idx>(max_iter));
}

inline solver_result field_solver::solve_var_poisson(scalar_field_3d &phi, const scalar_field_3d &coeff,
                                            const array<dirichlet_bc> &bcs, double tol,
                                            int max_iter) {
    const int nx = phi.nx(), ny = phi.ny(), nz = phi.nz();
    const idx N = phi.size();
    const double inv_dx2 = 1.0 / (phi.dx() * phi.dx());

    auto flat = [&](int i, int j, int k) -> idx {
        return static_cast<idx>((k * ny * nx) + (j * nx) + i);
    };

    table<int, double> bc_map;
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
    vec b(N, 0.0);
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

    auto matvec = [&](const vec &v, vec &Av) {
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
    return num::cg(A, b, phi.as_vec(), tol, static_cast<idx>(max_iter));
}

inline vector_field_3d field_solver::gradient(const scalar_field_3d &phi) {
    vector_field_3d out(phi.nx(), phi.ny(), phi.nz(), phi.dx(), phi.ox(), phi.oy(), phi.oz());
    gradient_3d(phi, out.x, out.y, out.z);
    return out;
}

inline scalar_field_3d field_solver::divergence(const vector_field_3d &f) {
    scalar_field_3d out(f.x.nx(), f.x.ny(), f.x.nz(), f.x.dx(), f.x.ox(), f.x.oy(), f.x.oz());
    divergence_3d(f.x, f.y, f.z, out);
    return out;
}

inline vector_field_3d field_solver::curl(const vector_field_3d &A) {
    vector_field_3d B(A.x.nx(), A.x.ny(), A.x.nz(), A.x.dx(), A.x.ox(), A.x.oy(), A.x.oz());
    curl_3d(A.x, A.y, A.z, B.x, B.y, B.z);
    return B;
}

// magnetic_solver

inline vector_field_3d magnetic_solver::current_density(const scalar_field_3d &sigma,
                                              const scalar_field_3d &phi) {
    vector_field_3d J = field_solver::gradient(phi);
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

inline vector_field_3d magnetic_solver::solve_magnetic_field(const vector_field_3d &J, double tol,
                                                   int max_iter) {
    const int nx = J.x.nx(), ny = J.x.ny(), nz = J.x.nz();
    const float dx = J.x.dx(), ox = J.x.ox(), oy = J.x.oy(), oz = J.x.oz();

    auto make_source = [&](const scalar_field_3d &Jc) {
        scalar_field_3d src(nx, ny, nz, dx, ox, oy, oz);
        for (int k = 0; k < nz; ++k) {
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    src(i, j, k) = -MU0 * Jc(i, j, k);
                }
            }
        }
        return src;
    };

    scalar_field_3d Ax(nx, ny, nz, dx, ox, oy, oz);
    scalar_field_3d Ay(nx, ny, nz, dx, ox, oy, oz);
    scalar_field_3d Az(nx, ny, nz, dx, ox, oy, oz);

    field_solver::solve_poisson(Ax, make_source(J.x), tol, max_iter);
    field_solver::solve_poisson(Ay, make_source(J.y), tol, max_iter);
    field_solver::solve_poisson(Az, make_source(J.z), tol, max_iter);

    vector_field_3d A(nx, ny, nz, dx, ox, oy, oz);
    A.x = Ax;
    A.y = Ay;
    A.z = Az;
    return field_solver::curl(A);
}

} // namespace num
