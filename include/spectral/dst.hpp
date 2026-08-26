/// @file spectral/dst.hpp
/// @brief Discrete sine transforms, built on the complex FFT.
///
/// The DST-I diagonalizes the Dirichlet Laplacian, which is what makes a direct
/// Poisson solve on a uniform grid \f$O(N^2 \log N)\f$ rather than \f$O(N^3)\f$.
/// It is expressed through the complex FFT by odd extension rather than being
/// implemented separately, so it inherits whichever FFT backend the build selected.
#pragma once

#include "container/vector.hpp"
#include "core/types.hpp"
#include "spectral/fft.hpp"
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace num::spectral {

/// @brief Reject a length the odd-extension route cannot transform.
///
/// The extension has length 2(N+1), and the radix-2 FFT needs that to be a
/// power of two, so N has to be one less than a power of two.
inline void dst_check_size(int N) {
    if (N <= 0 || (N & (N + 1)) != 0) {
        throw std::invalid_argument(
            "spectral::dst: N must equal 2^p - 1 (e.g. 7, 15, 31, 63, ...)");
    }
}

/// Unnormalised DST-I of an N-point vector via complex FFT.
///
/// X[k] = sum_{j=1}^{N} x[j] * sin(j*k*pi/(N+1)),  k = 1..N (stored 0-indexed).
/// Odd-extension y = [0, x, 0, -rev(x)] has length M = 2(N+1).
/// FFT(y)[k] = -2i * sum sin(...)  =>  DST(x)[k-1] = -Im(FFT(y)[k]) / 2.
inline Vector dst1(const Vector &x) {
    const int N = static_cast<int>(x.size());
    dst_check_size(N);
    const int M = 2 * (N + 1);
    CVector y(static_cast<std::size_t>(M), cplx{0.0, 0.0});
    for (int j = 0; j < N; ++j) {
        const auto sj = static_cast<std::size_t>(j);
        y[sj + 1] = cplx{x[sj], 0.0};
        y[static_cast<std::size_t>(M - 1 - j)] = cplx{-x[sj], 0.0};
    }
    CVector Y(static_cast<std::size_t>(M));
    spectral::fft(y, Y);
    Vector out(static_cast<std::size_t>(N));
    for (int k = 0; k < N; ++k) {
        out[static_cast<std::size_t>(k)] = -Y[static_cast<std::size_t>(k) + 1].imag() / 2.0;
    }
    return out;
}

inline void dst_rows(std::vector<double> &A, int N) {
    Vector row(static_cast<std::size_t>(N));
    for (int i = 0; i < N; ++i) {
        const std::size_t base = static_cast<std::size_t>(i) * static_cast<std::size_t>(N);
        for (int j = 0; j < N; ++j) {
            row[static_cast<std::size_t>(j)] = A[base + j];
        }
        row = dst1(row);
        for (int j = 0; j < N; ++j) {
            A[base + j] = row[static_cast<std::size_t>(j)];
        }
    }
}

inline void dst_cols(std::vector<double> &A, int N) {
    Vector col(static_cast<std::size_t>(N));
    for (int j = 0; j < N; ++j) {
        const std::size_t sj = static_cast<std::size_t>(j);
        for (int i = 0; i < N; ++i) {
            col[static_cast<std::size_t>(i)] =
                A[(static_cast<std::size_t>(i) * static_cast<std::size_t>(N)) + sj];
        }
        col = dst1(col);
        for (int i = 0; i < N; ++i) {
            A[(static_cast<std::size_t>(i) * static_cast<std::size_t>(N)) + sj] =
                col[static_cast<std::size_t>(i)];
        }
    }
}

inline void dst2d(std::vector<double> &A, int N) {
    dst_check_size(N);
    dst_cols(A, N);
    dst_rows(A, N);
}

} // namespace num::spectral
