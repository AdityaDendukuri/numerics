/// @file pde/stencil.hpp
/// @brief Higher-order stencil and grid-sweep utilities.
///
/// The second-order 2D stencil stores \f$h^2\Delta_h u\f$:
/// \f[
///   y_{ij}=u_{i+1,j}+u_{i-1,j}+u_{i,j+1}+u_{i,j-1}-4u_{ij}.
/// \f]
/// @todo Add boundary-condition-aware stencil operators and compact higher-
/// order derivative operators for gradient, divergence, curl, and Laplacian.
#pragma once

#include "core/types.hpp"
#include "container/vector.hpp"
#include "fields/field3d.hpp"
#include "fields/grid2d.hpp"
#include "fields/scalar_field_2d.hpp"
#include <algorithm>
#include <cmath>
#include <vector>

namespace num {

template <typename T>
void laplacian_stencil_2d(const basic_vec<T> &x, basic_vec<T> &y, int N) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            int k = (i * N) + j;
            T val = T(-4) * x[k];
            if (i > 0) {
                val += x[k - N];
            }
            if (i < N - 1) {
                val += x[k + N];
            }
            if (j > 0) {
                val += x[k - 1];
            }
            if (j < N - 1) {
                val += x[k + 1];
            }
            y[k] = val;
        }
    }
}

/// @brief Periodic second-order 2D Laplacian stencil.
template <typename T>
void laplacian_stencil_2d_periodic(const basic_vec<T> &x, basic_vec<T> &y, int N) {
    for (int i = 0; i < N; ++i) {
        int ip = (i + 1) % N, im = (i + N - 1) % N;
        const T *row = x.data() + (i * N);
        const T *row_p = x.data() + (ip * N);
        const T *row_m = x.data() + (im * N);
        T *d = y.data() + (i * N);

        d[0] = row_p[0] + row_m[0] + row[1] + row[N - 1] - (T(4) * row[0]);
        for (int j = 1; j < N - 1; ++j) {
            d[j] = row_p[j] + row_m[j] + row[j + 1] + row[j - 1] - (T(4) * row[j]);
        }
        d[N - 1] = row_p[N - 1] + row_m[N - 1] + row[0] + row[N - 2] - (T(4) * row[N - 1]);
    }
}

/// @brief Fourth-order 2D Laplacian cross stencil.
///
/// \f[
///   y_{ij} = \frac{1}{12}\bigl(
///     -x_{i-2,j} + 16x_{i-1,j} - 30x_{i,j} + 16x_{i+1,j} - x_{i+2,j}
///     -x_{i,j-2} + 16x_{i,j-1}              + 16x_{i,j+1} - x_{i,j+2}
///   \bigr)
/// \f]
template <typename T>
void laplacian_stencil_2d_4th(const basic_vec<T> &x, basic_vec<T> &y, int N) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            int k = (i * N) + j;
            T val = T(-30) * x[k];
            // i-axis: +/-1
            if (i > 0) {
                val += T(16) * x[((i - 1) * N) + j];
            }
            if (i < N - 1) {
                val += T(16) * x[((i + 1) * N) + j];
            }
            // i-axis: +/-2
            if (i > 1) {
                val -= x[((i - 2) * N) + j];
            }
            if (i < N - 2) {
                val -= x[((i + 2) * N) + j];
            }
            // j-axis: +/-1
            if (j > 0) {
                val += T(16) * x[k - 1];
            }
            if (j < N - 1) {
                val += T(16) * x[k + 1];
            }
            // j-axis: +/-2
            if (j > 1) {
                val -= x[k - 2];
            }
            if (j < N - 2) {
                val -= x[k + 2];
            }
            y[k] = val / T(12);
        }
    }
}

/// Bilinear interpolation on a periodic NxN grid with configurable stagger
/// offset.
///
/// field[i,j] is defined at physical position ((i + ox/h)*h, (j + oy/h)*h).
/// Returns the interpolated field value at physical point (px, py).
///
/// @param field Flattened grid samples.
/// @param N Grid width and height.
/// @param h Grid spacing.
/// @param px Physical x coordinate.
/// @param py Physical y coordinate.
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
inline real sample_2d_periodic(const vec &field, idx N, real h, real px, real py, real ox,
                               real oy) {
    real fx = std::fmod((px - ox) / h, static_cast<real>(N));
    real fy = std::fmod((py - oy) / h, static_cast<real>(N));
    if (fx < 0.0) {
        fx += N;
    }
    if (fy < 0.0) {
        fy += N;
    }
    idx i0 = static_cast<idx>(fx) % N;
    idx i1 = (i0 + 1) % N;
    real fi = fx - std::floor(fx);
    idx j0 = static_cast<idx>(fy) % N;
    idx j1 = (j0 + 1) % N;
    real fj = fy - std::floor(fy);
    return ((1 - fi) * (1 - fj) * field[(i0 * N) + j0]) + (fi * (1 - fj) * field[(i1 * N) + j0]) +
           ((1 - fi) * fj * field[(i0 * N) + j1]) + (fi * fj * field[(i1 * N) + j1]);
}

/// @brief Apply a mutable 1D operation to each column fiber.
template <typename T, typename F>
void col_fiber_sweep(basic_vec<T> &data, int N, F &&f) {
    array<T> fiber(N);
    for (int j = 0; j < N; ++j) {
        for (int i = 0; i < N; ++i) {
            fiber[i] = data[(i * N) + j];
        }
        f(fiber);
        for (int i = 0; i < N; ++i) {
            data[(i * N) + j] = fiber[i];
        }
    }
}

/// @brief Apply a mutable 1D operation to each row fiber.
template <typename T, typename F>
void row_fiber_sweep(basic_vec<T> &data, int N, F &&f) {
    array<T> fiber(N);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            fiber[j] = data[(i * N) + j];
        }
        f(fiber);
        for (int j = 0; j < N; ++j) {
            data[(i * N) + j] = fiber[j];
        }
    }
}

/// @brief Fill grid values at \f$x_i=(i+1)h,\ y_j=(j+1)h\f$.
template <typename F>
void fill_grid(vec &u, int N, double h, F &&f) {
    for (int i = 0; i < N; ++i) {
        double xi = (i + 1) * h;
        for (int j = 0; j < N; ++j) {
            u[(static_cast<std::size_t>(i) * N) + j] = f(xi, (j + 1) * h);
        }
    }
}

template <typename F>
void fill_grid(scalar_field_2d &g, F &&f) {
    fill_grid(g.as_vec(), g.N(), g.h(), std::forward<F>(f));
}

inline void laplacian_stencil_2d_periodic(const scalar_field_2d &x, scalar_field_2d &y) {
    laplacian_stencil_2d_periodic(x.as_vec(), y.as_vec(), x.N());
}

inline void laplacian_stencil_2d_4th(const scalar_field_2d &x, scalar_field_2d &y) {
    laplacian_stencil_2d_4th(x.as_vec(), y.as_vec(), x.N());
}

inline real sample_2d_periodic(const scalar_field_2d &g, real px, real py, real ox = 0.0,
                               real oy = 0.0) {
    return sample_2d_periodic(g.as_vec(), static_cast<idx>(g.N()), g.h(), px, py, ox, oy);
}

/// @brief Compute \f$-\Delta_h x\f$ on a 3D grid.
inline void neg_laplacian_3d(const vec &x, vec &y, int nx, int ny, int nz, double inv_dx2) {
    auto flat = [&](int i, int j, int k) -> idx {
        return static_cast<idx>((k * ny * nx) + (j * nx) + i);
    };
    auto interior_value = [&](int i, int j, int k) {
        if (i == 0 || i == nx - 1 || j == 0 || j == ny - 1 || k == 0 || k == nz - 1) {
            return 0.0;
        }
        return x[flat(i, j, k)];
    };
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                idx id = flat(i, j, k);
                if (i == 0 || i == nx - 1 || j == 0 || j == ny - 1 || k == 0 || k == nz - 1) {
                    y[id] = x[id];
                } else {
                    // Eliminate zero-Dirichlet boundary unknowns to keep the full operator
                    // symmetric.
                    y[id] = inv_dx2 * ((6.0 * x[id]) - interior_value(i + 1, j, k) -
                                       interior_value(i - 1, j, k) - interior_value(i, j + 1, k) -
                                       interior_value(i, j - 1, k) - interior_value(i, j, k + 1) -
                                       interior_value(i, j, k - 1));
                }
            }
        }
    }
}

/// @brief Compute \f$\nabla\phi\f$ with central differences.
inline void gradient_3d(const scalar_field_3d &phi, scalar_field_3d &gx, scalar_field_3d &gy,
                        scalar_field_3d &gz) {
    int nx = phi.nx(), ny = phi.ny(), nz = phi.nz();
    double inv2dx = 1.0 / (2.0 * phi.dx());
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                int ip = std::min(i + 1, nx - 1), im = std::max(i - 1, 0);
                int jp = std::min(j + 1, ny - 1), jm = std::max(j - 1, 0);
                int kp = std::min(k + 1, nz - 1), km = std::max(k - 1, 0);
                gx(i, j, k) = (phi(ip, j, k) - phi(im, j, k)) * inv2dx;
                gy(i, j, k) = (phi(i, jp, k) - phi(i, jm, k)) * inv2dx;
                gz(i, j, k) = (phi(i, j, kp) - phi(i, j, km)) * inv2dx;
            }
        }
    }
}

/// @brief Compute \f$\nabla\cdot f\f$ with central differences.
inline void divergence_3d(const scalar_field_3d &fx, const scalar_field_3d &fy, const scalar_field_3d &fz,
                          scalar_field_3d &out) {
    int nx = fx.nx(), ny = fx.ny(), nz = fx.nz();
    double inv2dx = 1.0 / (2.0 * fx.dx());
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                int ip = std::min(i + 1, nx - 1), im = std::max(i - 1, 0);
                int jp = std::min(j + 1, ny - 1), jm = std::max(j - 1, 0);
                int kp = std::min(k + 1, nz - 1), km = std::max(k - 1, 0);
                out(i, j, k) = ((fx(ip, j, k) - fx(im, j, k)) + (fy(i, jp, k) - fy(i, jm, k)) +
                                (fz(i, j, kp) - fz(i, j, km))) *
                               inv2dx;
            }
        }
    }
}

/// @brief Compute \f$\nabla\times A\f$ with central differences.
inline void curl_3d(const scalar_field_3d &ax, const scalar_field_3d &ay, const scalar_field_3d &az,
                    scalar_field_3d &bx, scalar_field_3d &by, scalar_field_3d &bz) {
    int nx = ax.nx(), ny = ax.ny(), nz = ax.nz();
    double inv2dx = 1.0 / (2.0 * ax.dx());
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                int ip = std::min(i + 1, nx - 1), im = std::max(i - 1, 0);
                int jp = std::min(j + 1, ny - 1), jm = std::max(j - 1, 0);
                int kp = std::min(k + 1, nz - 1), km = std::max(k - 1, 0);
                bx(i, j, k) = (az(i, jp, k) - az(i, jm, k) - ay(i, j, kp) + ay(i, j, km)) * inv2dx;
                by(i, j, k) = (ax(i, j, kp) - ax(i, j, km) - az(ip, j, k) + az(im, j, k)) * inv2dx;
                bz(i, j, k) = (ay(ip, j, k) - ay(im, j, k) - ax(i, jp, k) + ax(i, jm, k)) * inv2dx;
            }
        }
    }
}

} // namespace num
