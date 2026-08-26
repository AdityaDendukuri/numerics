/// @file small_matrix.hpp
/// @brief Constexpr fixed-size vector, matrix, and Givens rotation helpers.
#pragma once

#include "container/vector_ops.hpp"

#include "core/types.hpp"
#include <array>
#include <cmath>

namespace num {

template <idx N>
/// Stack-allocated fixed-size real vector with constexpr arithmetic.
struct SmallVec {
    std::array<real, N> data{};

    constexpr real &operator[](idx i) noexcept { return data[i]; }
    constexpr const real &operator[](idx i) const noexcept { return data[i]; }
    static constexpr idx size() noexcept { return N; }

    constexpr SmallVec &operator+=(const SmallVec &o) noexcept {
        for (idx i = 0; i < N; ++i) {
            data[i] += o.data[i];
        }
        return *this;
    }

    constexpr SmallVec &operator-=(const SmallVec &o) noexcept {
        for (idx i = 0; i < N; ++i) {
            data[i] -= o.data[i];
        }
        return *this;
    }

    constexpr SmallVec &operator*=(real s) noexcept {
        for (idx i = 0; i < N; ++i) {
            data[i] *= s;
        }
        return *this;
    }

    constexpr real dot(const SmallVec &o) const noexcept {
        real s = 0;
        for (idx i = 0; i < N; ++i) {
            s += data[i] * o.data[i];
        }
        return s;
    }

    [[nodiscard]] constexpr real norm_sq() const noexcept { return dot(*this); }
};

template <idx N>
constexpr SmallVec<N> operator+(SmallVec<N> a, const SmallVec<N> &b) noexcept {
    return a += b;
}

template <idx N>
constexpr SmallVec<N> operator*(real s, SmallVec<N> v) noexcept {
    return v *= s;
}

template <idx M, idx N>
/// Stack-allocated row-major real matrix with constexpr arithmetic.
struct SmallMatrix {
    std::array<real, M * N> data{};

    constexpr real &operator()(idx i, idx j) noexcept { return data[(i * N) + j]; }

    constexpr const real &operator()(idx i, idx j) const noexcept { return data[(i * N) + j]; }

    static constexpr idx rows() noexcept { return M; }
    static constexpr idx cols() noexcept { return N; }

    constexpr void fill(real v) noexcept { data.fill(v); }

    /// Construct a zero-filled matrix.
    static constexpr SmallMatrix zeros() noexcept { return SmallMatrix{}; }

    /// Construct an identity matrix; available only for square shapes.
    static constexpr SmallMatrix identity() noexcept {
        static_assert(M == N, "identity() requires a square matrix");
        SmallMatrix m{};
        for (idx i = 0; i < M; ++i) {
            m(i, i) = 1;
        }
        return m;
    }

    /// Return a copy with rows and columns exchanged.
    constexpr SmallMatrix<N, M> transposed() const noexcept {
        SmallMatrix<N, M> t{};
        for (idx i = 0; i < M; ++i) {
            for (idx j = 0; j < N; ++j) {
                t(j, i) = (*this)(i, j);
            }
        }
        return t;
    }

    template <idx K>
    constexpr SmallMatrix<M, K> operator*(const SmallMatrix<N, K> &B) const noexcept {
        SmallMatrix<M, K> C{};
        for (idx i = 0; i < M; ++i) {
            for (idx k = 0; k < N; ++k) {
                for (idx j = 0; j < K; ++j) {
                    C(i, j) += (*this)(i, k) * B(k, j);
                }
            }
        }
        return C;
    }

    constexpr SmallVec<M> operator*(const SmallVec<N> &x) const noexcept {
        SmallVec<M> y{};
        for (idx i = 0; i < M; ++i) {
            for (idx j = 0; j < N; ++j) {
                y[i] += (*this)(i, j) * x[j];
            }
        }
        return y;
    }

    constexpr SmallMatrix &operator+=(const SmallMatrix &o) noexcept {
        for (idx k = 0; k < M * N; ++k) {
            data[k] += o.data[k];
        }
        return *this;
    }

    constexpr SmallMatrix &operator*=(real s) noexcept {
        for (idx k = 0; k < M * N; ++k) {
            data[k] *= s;
        }
        return *this;
    }
};

/// Plane rotation used to eliminate one component of a two-vector.
struct GivensRotation {
    real c = 1;
    real s = 0;

    /// @brief Construct \f$G=\begin{bmatrix}c&s\\-s&c\end{bmatrix}\f$.
    static constexpr GivensRotation from(real a, real b) noexcept {
        if (b == 0) {
            return {};
        }
        const real r = std::sqrt((a * a) + (b * b));
        return {a / r, b / r};
    }

    /// Apply G to the pair (x,y) in place.
    constexpr void apply(real &x, real &y) const noexcept {
        const real tmp = (c * x) + (s * y);
        y = (-s * x) + (c * y);
        x = tmp;
    }

    /// Apply G^T to the pair (x,y) in place.
    constexpr void apply_t(real &x, real &y) const noexcept {
        const real tmp = (c * x) - (s * y);
        y = (s * x) + (c * y);
        x = tmp;
    }

    /// Materialize the rotation as a 2x2 matrix.
    constexpr SmallMatrix<2, 2> matrix() const noexcept {
        SmallMatrix<2, 2> G{};
        G(0, 0) = c;
        G(0, 1) = s;
        G(1, 0) = -s;
        G(1, 1) = c;
        return G;
    }
};

} // namespace num
