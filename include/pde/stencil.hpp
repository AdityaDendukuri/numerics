/// @file pde/stencil.hpp
/// @brief Higher-order stencil and grid-sweep utilities.
///
/// 2D (NxN grid, row-major storage: index = i*N + j, Dirichlet or periodic BC):
///   laplacian_stencil_2d           -- y = (sum of 4 nbrs) - 4x, Dirichlet
///   laplacian_stencil_2d_periodic  -- same, periodic wrap (boundary-peeled for
///   vectorization) sample_2d_periodic             -- bilinear interpolation
///   with stagger offset col_fiber_sweep                -- for each column j:
///   extract fiber, call f, write back row_fiber_sweep                -- for
///   each row    i: extract fiber, call f, write back
///
/// 3D (Grid3D, central-difference, one-sided at boundaries):
///   neg_laplacian_3d   -- y = -Lap(x) (Dirichlet: boundary rows = identity)
///   gradient_3d        -- (gx,gy,gz) = grad(phi)
///   divergence_3d      -- div f = d(fx)/dx + d(fy)/dy + d(fz)/dz
///   curl_3d            -- curl(A)
#pragma once

#include "core/vector.hpp"
#include "spatial/grid3d.hpp"
#include <vector>
#include <algorithm>
#include <cmath>

namespace num {

// 2-D stencils  (NxN, row-major: data[i*N + j])

/// Compute y[i,j] = x[i+1,j] + x[i-1,j] + x[i,j+1] + x[i,j-1] - 4*x[i,j]///
/// Dirichlet: out-of-bounds neighbors contribute 0. Works for T = real or cplx.
template<typename T>
void laplacian_stencil_2d(const BasicVector<T>& x, BasicVector<T>& y, int N) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            int k   = i * N + j;
            T   val = T(-4) * x[k];
            if (i > 0)
                val += x[k - N];
            if (i < N - 1)
                val += x[k + N];
            if (j > 0)
                val += x[k - 1];
            if (j < N - 1)
                val += x[k + 1];
            y[k] = val;
        }
    }
}

/// Same as laplacian_stencil_2d but with periodic (wrap-around) boundaries.
///
/// The j boundaries (j=0 and j=N-1) are peeled out of the inner loop so the
/// interior range j=1..N-2 contains no modulo arithmetic and is
/// auto-vectorizable. Row wrapping (i+/-1) uses precomputed indices.
template<typename T>
void laplacian_stencil_2d_periodic(const BasicVector<T>& x,
                                   BasicVector<T>&       y,
                                   int                   N) {
    for (int i = 0; i < N; ++i) {
        int      ip = (i + 1) % N, im = (i + N - 1) % N;
        const T* row   = x.data() + i * N;
        const T* row_p = x.data() + ip * N;
        const T* row_m = x.data() + im * N;
        T*       d     = y.data() + i * N;

        // j = 0: left neighbor wraps to j = N-1
        d[0] = row_p[0] + row_m[0] + row[1] + row[N - 1] - T(4) * row[0];
        // j = 1..N-2: no wrap, compiler can vectorise
        for (int j = 1; j < N - 1; ++j)
            d[j] = row_p[j] + row_m[j] + row[j + 1] + row[j - 1]
                   - T(4) * row[j];
        // j = N-1: right neighbor wraps to j = 0
        d[N - 1] = row_p[N - 1] + row_m[N - 1] + row[0] + row[N - 2]
                   - T(4) * row[N - 1];
    }
}

/// Bilinear interpolation on a periodic NxN grid with configurable stagger
/// offset.
///
/// field[i,j] is defined at physical position ((i + ox/h)*h, (j + oy/h)*h).
/// Returns the interpolated field value at physical point (px, py).
///
/// @param ox  x-axis origin offset in physical units (0 for unstaggered, h/2
/// for v-face)
/// @param oy  y-axis origin offset in physical units (0 for unstaggered, h/2
/// for u-face)
///
/// MAC grid usage:
/// \f[
///   \text{interp\_u}(px,py) = \texttt{sample\_2d\_periodic}(u, N, h,\; px,
///   py,\; 0,\; h/2)
/// \f]
/// \f[
///   \text{interp\_v}(px,py) = \texttt{sample\_2d\_periodic}(v, N, h,\; px,
///   py,\; h/2,\; 0)
/// \f]
inline real sample_2d_periodic(const Vector& field,
                               idx           N,
                               real          h,
                               real          px,
                               real          py,
                               real          ox,
                               real          oy) {
    real fx = std::fmod((px - ox) / h, static_cast<real>(N));
    real fy = std::fmod((py - oy) / h, static_cast<real>(N));
    if (fx < 0.0)
        fx += N;
    if (fy < 0.0)
        fy += N;
    idx  i0 = static_cast<idx>(fx) % N;
    idx  i1 = (i0 + 1) % N;
    real fi = fx - std::floor(fx);
    idx  j0 = static_cast<idx>(fy) % N;
    idx  j1 = (j0 + 1) % N;
    real fj = fy - std::floor(fy);
    return (1 - fi) * (1 - fj) * field[i0 * N + j0]
           + fi * (1 - fj) * field[i1 * N + j0]
           + (1 - fi) * fj * field[i0 * N + j1] + fi * fj * field[i1 * N + j1];
}

/// For each column j in [0,N), extract the x-direction fiber
/// data[0..N-1, j] into a std::vector<T>, call f(fiber), then write back.
///
/// f must have signature: void(std::vector<T>&) and modifies in-place.
///
/// Use for ADI / Crank-Nicolson sweeps along x.
template<typename T, typename F>
void col_fiber_sweep(BasicVector<T>& data, int N, F&& f) {
    std::vector<T> fiber(N);
    for (int j = 0; j < N; ++j) {
        for (int i = 0; i < N; ++i)
            fiber[i] = data[i * N + j];
        f(fiber);
        for (int i = 0; i < N; ++i)
            data[i * N + j] = fiber[i];
    }
}

/// For each row i in [0,N), extract the y-direction fiber
/// data[i, 0..N-1] into a std::vector<T>, call f(fiber), then write back.
///
/// Use for ADI / Crank-Nicolson sweeps along y.
template<typename T, typename F>
void row_fiber_sweep(BasicVector<T>& data, int N, F&& f) {
    std::vector<T> fiber(N);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j)
            fiber[j] = data[i * N + j];
        f(fiber);
        for (int j = 0; j < N; ++j)
            data[i * N + j] = fiber[j];
    }
}

// 3-D stencils  (Grid3D)

/// Compute the negative Laplacian: y = -Lap(x) = (6x[i,j,k] - sum of 6 nbrs) /
/// dx^2 Dirichlet BC: boundary nodes satisfy y[bdry] = x[bdry] (identity row,
/// so that the system is SPD when used with a b=0 boundary RHS).
///
/// x and y are flat vectors in Grid3D layout: idx = k*ny*nx + j*nx + i.
inline void neg_laplacian_3d(const Vector& x,
                             Vector&       y,
                             int           nx,
                             int           ny,
                             int           nz,
                             double        inv_dx2) {
    auto flat = [&](int i, int j, int k) -> idx {
        return static_cast<idx>(k * ny * nx + j * nx + i);
    };
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                idx id = flat(i, j, k);
                if (i == 0 || i == nx - 1 || j == 0 || j == ny - 1 || k == 0
                    || k == nz - 1) {
                    y[id] = x[id];
                } else {
                    y[id] = inv_dx2
                            * (6.0 * x[id] - x[flat(i + 1, j, k)]
                               - x[flat(i - 1, j, k)] - x[flat(i, j + 1, k)]
                               - x[flat(i, j - 1, k)] - x[flat(i, j, k + 1)]
                               - x[flat(i, j, k - 1)]);
                }
            }
}

/// Compute the central-difference gradient of a scalar field./// One-sided
/// differences at domain boundaries. gx, gy, gz must already be allocated with
/// the same dimensions as phi.
inline void gradient_3d(const Grid3D& phi, Grid3D& gx, Grid3D& gy, Grid3D& gz) {
    int    nx = phi.nx(), ny = phi.ny(), nz = phi.nz();
    double inv2dx = 1.0 / (2.0 * phi.dx());
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                int ip = std::min(i + 1, nx - 1), im = std::max(i - 1, 0);
                int jp = std::min(j + 1, ny - 1), jm = std::max(j - 1, 0);
                int kp = std::min(k + 1, nz - 1), km = std::max(k - 1, 0);
                gx(i, j, k) = (phi(ip, j, k) - phi(im, j, k)) * inv2dx;
                gy(i, j, k) = (phi(i, jp, k) - phi(i, jm, k)) * inv2dx;
                gz(i, j, k) = (phi(i, j, kp) - phi(i, j, km)) * inv2dx;
            }
}

/// Compute the central-difference divergence of a vector field (fx, fy, fz).
/// One-sided differences at domain boundaries.
/// out must already be allocated with the same dimensions.
inline void divergence_3d(const Grid3D& fx,
                          const Grid3D& fy,
                          const Grid3D& fz,
                          Grid3D&       out) {
    int    nx = fx.nx(), ny = fx.ny(), nz = fx.nz();
    double inv2dx = 1.0 / (2.0 * fx.dx());
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                int ip = std::min(i + 1, nx - 1), im = std::max(i - 1, 0);
                int jp = std::min(j + 1, ny - 1), jm = std::max(j - 1, 0);
                int kp = std::min(k + 1, nz - 1), km = std::max(k - 1, 0);
                out(i, j, k) = ((fx(ip, j, k) - fx(im, j, k))
                                + (fy(i, jp, k) - fy(i, jm, k))
                                + (fz(i, j, kp) - fz(i, j, km)))
                               * inv2dx;
            }
}

/// Compute the central-difference curl(A) of a vector field.
/// One-sided differences at domain boundaries.
/// bx, by, bz must already be allocated with the same dimensions as ax, ay, az.
inline void curl_3d(const Grid3D& ax,
                    const Grid3D& ay,
                    const Grid3D& az,
                    Grid3D&       bx,
                    Grid3D&       by,
                    Grid3D&       bz) {
    int    nx = ax.nx(), ny = ax.ny(), nz = ax.nz();
    double inv2dx = 1.0 / (2.0 * ax.dx());
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                int ip = std::min(i + 1, nx - 1), im = std::max(i - 1, 0);
                int jp = std::min(j + 1, ny - 1), jm = std::max(j - 1, 0);
                int kp = std::min(k + 1, nz - 1), km = std::max(k - 1, 0);
                bx(i, j, k) = (az(i, jp, k) - az(i, jm, k) - ay(i, j, kp)
                               + ay(i, j, km))
                              * inv2dx;
                by(i, j, k) = (ax(i, j, kp) - ax(i, j, km) - az(ip, j, k)
                               + az(im, j, k))
                              * inv2dx;
                bz(i, j, k) = (ay(ip, j, k) - ay(im, j, k) - ax(i, jp, k)
                               + ax(i, jm, k))
                              * inv2dx;
            }
}

} // namespace num
