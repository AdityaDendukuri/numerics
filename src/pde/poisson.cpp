/// @file pde/poisson.cpp
#include "pde/poisson.hpp"
#include "spectral/fft.hpp"
#include "core/vector.hpp"
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace num {
namespace pde {

namespace {

// Unnormalised DST-I of an N-point vector via complex FFT.
//
// X[k] = sum_{j=1}^{N} x[j] * sin(j*k*pi/(N+1)),  k = 1..N (stored 0-indexed).
// Odd-extension y = [0, x, 0, -rev(x)] has length M = 2(N+1).
// FFT(y)[k] = -2i * sum sin(...)  =>  DST(x)[k-1] = -Im(FFT(y)[k]) / 2.
static Vector dst1(const Vector& x) {
    const int N = static_cast<int>(x.size());
    const int M = 2 * (N + 1);
    CVector y(static_cast<std::size_t>(M), cplx{0.0, 0.0});
    for (int j = 0; j < N; ++j) {
        const auto sj = static_cast<std::size_t>(j);
        y[sj + 1]                               = cplx{ x[sj], 0.0};
        y[static_cast<std::size_t>(M - 1 - j)]  = cplx{-x[sj], 0.0};
    }
    CVector Y(static_cast<std::size_t>(M));
    spectral::fft(y, Y);
    Vector out(static_cast<std::size_t>(N));
    for (int k = 0; k < N; ++k) {
        out[static_cast<std::size_t>(k)] = -Y[static_cast<std::size_t>(k) + 1].imag() / 2.0;
    }
    return out;
}

static void dst_rows(std::vector<double>& A, int N) {
    Vector row(static_cast<std::size_t>(N));
    for (int i = 0; i < N; ++i) {
        const std::size_t base = static_cast<std::size_t>(i) * static_cast<std::size_t>(N);
        for (int j = 0; j < N; ++j) { row[static_cast<std::size_t>(j)] = A[base + j]; }
        row = dst1(row);
        for (int j = 0; j < N; ++j) { A[base + j] = row[static_cast<std::size_t>(j)]; }
    }
}

static void dst_cols(std::vector<double>& A, int N) {
    Vector col(static_cast<std::size_t>(N));
    for (int j = 0; j < N; ++j) {
        const std::size_t sj = static_cast<std::size_t>(j);
        for (int i = 0; i < N; ++i) {
            col[static_cast<std::size_t>(i)] = A[(static_cast<std::size_t>(i) * static_cast<std::size_t>(N)) + sj];
        }
        col = dst1(col);
        for (int i = 0; i < N; ++i) {
            A[(static_cast<std::size_t>(i) * static_cast<std::size_t>(N)) + sj] = col[static_cast<std::size_t>(i)];
        }
    }
}

// 2-D DST-I:  F^T * A * F  (columns first, then rows).
static void dst2d(std::vector<double>& A, int N) {
    dst_cols(A, N);
    dst_rows(A, N);
}

static void check_n(int N) {
    if (N <= 0 || (N & (N + 1)) != 0) {
        throw std::invalid_argument(
            "pde::poisson2d: N must equal 2^p - 1 (e.g. 7, 15, 31, 63, ...)");
    }
}

static std::vector<double> flatten(const Matrix& M, int N) {
    std::vector<double> v(static_cast<std::size_t>(N) * static_cast<std::size_t>(N));
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            v[(static_cast<std::size_t>(i) * static_cast<std::size_t>(N)) + static_cast<std::size_t>(j)] =
                M(static_cast<idx>(i), static_cast<idx>(j));
        }
    }
    return v;
}

static Matrix unflatten(const std::vector<double>& v, int N) {
    Matrix M(static_cast<idx>(N), static_cast<idx>(N));
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            M(static_cast<idx>(i), static_cast<idx>(j)) =
                v[(static_cast<std::size_t>(i) * static_cast<std::size_t>(N)) + static_cast<std::size_t>(j)];
        }
    }
    return M;
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// FD solver: eigenvalues D_k = 2(1 - cos(k*pi/(N+1))).  Error O(h^2).
//
//   1. f_hat = DST2D(h^2 * f)
//   2. u_hat_{jk} = f_hat_{jk} / (D_j + D_k)
//   3. u = (2/(N+1))^2 * DST2D(u_hat)   [IDST-I = (2/(N+1)) * DST-I]
// ---------------------------------------------------------------------------
Matrix poisson2d_fd(const Matrix& f, int N) {
    check_n(N);
    const double h  = 1.0 / (N + 1);
    const double pi = M_PI;

    std::vector<double> buf = flatten(f, N);
    const std::size_t NN = static_cast<std::size_t>(N) * static_cast<std::size_t>(N);
    for (std::size_t k = 0; k < NN; ++k) { buf[k] *= h * h; }
    dst2d(buf, N);

    std::vector<double> lam(static_cast<std::size_t>(N));
    for (int k = 0; k < N; ++k) {
        lam[static_cast<std::size_t>(k)] = 2.0 * (1.0 - std::cos((k + 1) * pi / (N + 1)));
    }
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            buf[(static_cast<std::size_t>(i) * static_cast<std::size_t>(N)) + static_cast<std::size_t>(j)] /=
                lam[static_cast<std::size_t>(i)] + lam[static_cast<std::size_t>(j)];
        }
    }
    dst2d(buf, N);
    const double s = (2.0 / (N + 1)) * (2.0 / (N + 1));
    for (double& v : buf) { v *= s; }

    return unflatten(buf, N);
}

// ---------------------------------------------------------------------------
// Spectral solver: exact eigenvalues (k*pi)^2.
//
//   1. f_hat = DST2D(f)
//   2. u_hat_{jk} = [4 / ((N+1)^2 * pi^2 * (j^2+k^2))] * f_hat_{jk}
//   3. u = DST2D(u_hat)
//
// DST satisfies F*F = ((N+1)/2)*I, so the continuous spectral coefficient is
// f_tilde_{jk} = (4/(N+1)^2) * f_hat_{j-1,k-1}.  Dividing by (j^2+k^2)*pi^2
// and reconstructing via DST gives machine-precision error for any f whose
// DST representation is exact on the N-point grid.
// ---------------------------------------------------------------------------
Matrix poisson2d(const Matrix& f, int N) {
    check_n(N);
    const double pi   = M_PI;
    const double N1sq = static_cast<double>(N + 1) * (N + 1);

    std::vector<double> buf = flatten(f, N);
    dst2d(buf, N);

    for (int i = 0; i < N; ++i) {
        const double ji = i + 1;
        for (int j = 0; j < N; ++j) {
            const double jj = j + 1;
            buf[(static_cast<std::size_t>(i) * static_cast<std::size_t>(N)) + static_cast<std::size_t>(j)] *=
                4.0 / (N1sq * pi * pi * ((ji * ji) + (jj * jj)));
        }
    }
    dst2d(buf, N);

    return unflatten(buf, N);
}

} // namespace pde
} // namespace num
