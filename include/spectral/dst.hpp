/// @file spectral/dst.hpp
/// @brief Discrete sine transforms through the complex FFT.
#pragma once

#include "container/vector.hpp"
#include "core/types.hpp"
#include "spectral/fft.hpp"
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace num::spectral {

/// @brief DST length check: \f$N + 1\f$ must be a power of two.
inline void dst_check_size(int N) {
    if (N <= 0 || (N & (N + 1)) != 0) {
        throw std::invalid_argument(
            "spectral::dst: N must equal 2^p - 1 (e.g. 7, 15, 31, 63, ...)");
    }
}

/// @brief Discrete sine transform (DST-I): \f$X_k = \sum_{j=1}^N x_j \sin(j k \pi / (N + 1))\f$.
/// Unnormalized.
/// @note \f$N + 1\f$ must be a power of two.
inline vec<real> dst1(const vec<real> &x) {
    const int N = static_cast<int>(x.size());
    dst_check_size(N);
    const int M = 2 * (N + 1);
    vec<cplx> y(static_cast<std::size_t>(M), cplx{0.0, 0.0});
    for (int j = 0; j < N; ++j) {
        const auto sj = static_cast<std::size_t>(j);
        y[sj + 1] = cplx{x[sj], 0.0};
        y[static_cast<std::size_t>(M - 1 - j)] = cplx{-x[sj], 0.0};
    }
    vec<cplx> Y(static_cast<std::size_t>(M));
    spectral::fft(y, Y);
    vec<real> out(static_cast<std::size_t>(N));
    for (int k = 0; k < N; ++k) {
        out[static_cast<std::size_t>(k)] = -Y[static_cast<std::size_t>(k) + 1].imag() / 2.0;
    }
    return out;
}

/// @brief Row-wise DST-I of an \f$N \times N\f$ array, in place.
inline void dst_rows(array<double> &A, int N) {
    vec<real> row(static_cast<std::size_t>(N));
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

/// @brief Column-wise DST-I of an \f$N \times N\f$ array, in place.
inline void dst_cols(array<double> &A, int N) {
    vec<real> col(static_cast<std::size_t>(N));
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

/// @brief 2D DST-I of an \f$N \times N\f$ array, in place.
inline void dst2d(array<double> &A, int N) {
    dst_check_size(N);
    dst_cols(A, N);
    dst_rows(A, N);
}

} // namespace num::spectral
