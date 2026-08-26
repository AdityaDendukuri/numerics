/// @file pde/poisson.cpp
#include "pde/poisson.hpp"
#include "spectral/dst.hpp"
#include "container/vector.hpp"
#include "spectral/fft.hpp"
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace num {
namespace pde {

namespace {

// X[k] = sum_{j=1}^{N} x[j] * sin(j*k*pi/(N+1)),  k = 1..N (stored 0-indexed).
// Odd-extension y = [0, x, 0, -rev(x)] has length M = 2(N+1).
// FFT(y)[k] = -2i * sum sin(...)  =>  DST(x)[k-1] = -Im(FFT(y)[k]) / 2.



// 2-D DST-I:  F^T * A * F  (columns first, then rows).

static void check_n(int N) {
    if (N <= 0 || (N & (N + 1)) != 0) {
        throw std::invalid_argument(
            "pde::poisson2d: N must equal 2^p - 1 (e.g. 7, 15, 31, 63, ...)");
    }
}

static std::vector<double> flatten(const Matrix &M, int N) {
    std::vector<double> v(static_cast<std::size_t>(N) * static_cast<std::size_t>(N));
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            v[(static_cast<std::size_t>(i) * static_cast<std::size_t>(N)) +
              static_cast<std::size_t>(j)] = M(static_cast<idx>(i), static_cast<idx>(j));
        }
    }
    return v;
}

static Matrix unflatten(const std::vector<double> &v, int N) {
    Matrix M(static_cast<idx>(N), static_cast<idx>(N));
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            M(static_cast<idx>(i), static_cast<idx>(j)) =
                v[(static_cast<std::size_t>(i) * static_cast<std::size_t>(N)) +
                  static_cast<std::size_t>(j)];
        }
    }
    return M;
}

} // anonymous namespace

Matrix poisson2d_fd(const Matrix &f, int N) {
    check_n(N);
    const double h = 1.0 / (N + 1);
    const double pi = M_PI;

    std::vector<double> buf = flatten(f, N);
    const std::size_t NN = static_cast<std::size_t>(N) * static_cast<std::size_t>(N);
    for (std::size_t k = 0; k < NN; ++k) {
        buf[k] *= h * h;
    }
    spectral::dst2d(buf, N);

    std::vector<double> lam(static_cast<std::size_t>(N));
    for (int k = 0; k < N; ++k) {
        lam[static_cast<std::size_t>(k)] = 2.0 * (1.0 - std::cos((k + 1) * pi / (N + 1)));
    }
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            buf[(static_cast<std::size_t>(i) * static_cast<std::size_t>(N)) +
                static_cast<std::size_t>(j)] /=
                lam[static_cast<std::size_t>(i)] + lam[static_cast<std::size_t>(j)];
        }
    }
    spectral::dst2d(buf, N);
    const double s = (2.0 / (N + 1)) * (2.0 / (N + 1));
    for (double &v : buf) {
        v *= s;
    }

    return unflatten(buf, N);
}

Matrix poisson2d(const Matrix &f, int N) {
    check_n(N);
    const double pi = M_PI;
    const double N1sq = static_cast<double>(N + 1) * (N + 1);

    std::vector<double> buf = flatten(f, N);
    spectral::dst2d(buf, N);

    for (int i = 0; i < N; ++i) {
        const double ji = i + 1;
        for (int j = 0; j < N; ++j) {
            const double jj = j + 1;
            buf[(static_cast<std::size_t>(i) * static_cast<std::size_t>(N)) +
                static_cast<std::size_t>(j)] *= 4.0 / (N1sq * pi * pi * ((ji * ji) + (jj * jj)));
        }
    }
    spectral::dst2d(buf, N);

    return unflatten(buf, N);
}

} // namespace pde
} // namespace num
