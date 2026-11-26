/// @file kernel/vector.hpp
/// @brief Level-1 kernels over raw pointers: vector updates, reductions, norms, rotations.
///
/// SPDX-License-Identifier: MIT
/// Part of numerics, (c) 2026 Aditya Dendukuri.
/// https://github.com/AdityaDendukuri/numerics
///
/// Standard library only; keep the attribution lines when copying.
///
/// @section kernel_contract Kernel contract
///
/// - Preconditions are unchecked; violating one is undefined behaviour.
/// - Buffers are caller-allocated; no kernel allocates or keeps a pointer.
/// - `NUM_K_RESTRICT` pointers must not overlap unless the function says they may alias.
/// - Everything is `noexcept`; numerical failure is returned, never thrown.
/// - Reductions reorder sums for speed; `contract::ordered` sums in index order.
#pragma once

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstddef>

namespace num {
/// @brief Index type, as in core/types.hpp.
using idx = std::size_t;
} // namespace num

#if defined(__GNUC__) || defined(__clang__)
#define NUM_K_AINLINE [[gnu::always_inline]] inline
#define NUM_K_RESTRICT __restrict__
#define NUM_K_IVDEP _Pragma("GCC ivdep")
// GCC/clang vector extensions: register-wide accumulators for the reductions.
#define NUM_K_VECTOR_EXT 1
#else
#define NUM_K_AINLINE inline
#define NUM_K_RESTRICT
#define NUM_K_IVDEP
#endif

// Bytes in the target's widest vector register.
#if defined(__AVX512F__)
#define NUM_K_VECTOR_BYTES 64
#elif defined(__AVX__)
#define NUM_K_VECTOR_BYTES 32
#else
#define NUM_K_VECTOR_BYTES 16
#endif

// Vector registers on the target; sizes gemm's register tile so it never spills.
#if defined(__AVX512F__) || defined(__aarch64__) || defined(_M_ARM64)
#define NUM_K_VECTOR_REGISTERS 32
#else
#define NUM_K_VECTOR_REGISTERS 16
#endif

// Cache sizes for gemm's blocking. Conservative defaults; CMake sets the host's sizes, or
// define them before including this file.
#ifndef NUM_K_L1_BYTES
#define NUM_K_L1_BYTES 32768
#endif
#ifndef NUM_K_L2_BYTES
#define NUM_K_L2_BYTES 524288
#endif
// B panel gemm keeps resident per sweep of rows; 1 MiB measured best.
#ifndef NUM_K_GEMM_PANEL_BYTES
#define NUM_K_GEMM_PANEL_BYTES 1048576
#endif

// Whether a*b + c is one fused instruction. x86 needs -mfma; GCC also -ffp-contract=fast.
#if defined(__FMA__) || defined(__ARM_FEATURE_FMA) || defined(__aarch64__) || defined(_M_ARM64)
#define NUM_K_HAS_FMA 1
#else
#define NUM_K_HAS_FMA 0
#endif

namespace num::kernel {

/// @brief Tags that relax or tighten the kernel contract at a call site.
namespace contract {
struct alias_safe_t final {};
struct throughput_t final {};
struct ordered_t final {};

/// @brief Allow the output to alias an input.
inline constexpr alias_safe_t alias_safe{};
/// @brief Allow sums to be reordered for speed.
inline constexpr throughput_t throughput{};
/// @brief Require sums in index order.
inline constexpr ordered_t ordered{};
} // namespace contract

namespace detail {

/// @brief Number of independent running sums in a reduction.
inline constexpr idx reduction_lanes = 8;

#if defined(NUM_K_VECTOR_EXT)

/// @brief Number of vector running sums in a reduction, when vector extensions are available.
inline constexpr idx reduction_accumulators = 4;

/// @brief Native vector type of `Bytes` bytes holding `T`.
template <class T, idx Bytes>
struct vector_of {
    typedef T type __attribute__((vector_size(Bytes)));
};

#endif

/// @brief Sum `element(i)` over \f$i < n\f$ in blocks: \f$\sum_{i<n} \mathrm{element}(i)\f$.
/// Visits each index once, so `element` may have side effects.
/// @note The summation order depends on the vector width; use `contract::ordered` for
/// results reproducible across machines.
template <std::floating_point T, class Element>
NUM_K_AINLINE T reduce(idx n, Element element) noexcept {
    idx i = 0;
    T total = T(0);

#if defined(NUM_K_VECTOR_EXT)
    using vector = typename vector_of<T, NUM_K_VECTOR_BYTES>::type;
    constexpr idx width = NUM_K_VECTOR_BYTES / sizeof(T);
    constexpr idx accumulators = reduction_accumulators;
    constexpr idx block = width * accumulators;

    vector lane[accumulators]{};
    for (; i + block <= n; i += block) {
        for (idx a = 0; a < accumulators; ++a) {
            vector value{};
            for (idx k = 0; k < width; ++k) {
                value[k] = element(i + (a * width) + k);
            }
            lane[a] += value;
        }
    }
    // One vector at a time, so the scalar tail is shorter than a vector.
    for (; i + width <= n; i += width) {
        vector value{};
        for (idx k = 0; k < width; ++k) {
            value[k] = element(i + k);
        }
        lane[0] += value;
    }
    for (idx w = accumulators / 2; w > 0; w /= 2) {
        for (idx a = 0; a < w; ++a) {
            lane[a] += lane[a + w];
        }
    }
    for (idx k = 0; k < width; ++k) {
        total += lane[0][k];
    }
#else
    // Portable fallback: independent scalar chains.
    constexpr idx lanes = reduction_lanes;
    T lane[lanes]{};
    for (; i + lanes <= n; i += lanes) {
        NUM_K_IVDEP
        for (idx k = 0; k < lanes; ++k) {
            lane[k] += element(i + k);
        }
    }
    for (idx w = lanes / 2; w > 0; w /= 2) {
        for (idx k = 0; k < w; ++k) {
            lane[k] += lane[k + w];
        }
    }
    total = lane[0];
#endif

    for (; i < n; ++i) {
        total += element(i);
    }
    return total;
}

} // namespace detail

/// @brief Copy a vector: \f$y \leftarrow x\f$.
template <typename T>
NUM_K_AINLINE void copy(T *NUM_K_RESTRICT y, const T *NUM_K_RESTRICT x, idx n) noexcept {
    NUM_K_IVDEP
    for (idx i = 0; i < n; ++i) {
        y[i] = x[i];
    }
}

/// @brief Set every entry to `value`: \f$x_i \leftarrow \mathrm{value}\f$.
template <typename T>
NUM_K_AINLINE void fill(T *NUM_K_RESTRICT x, T value, idx n) noexcept {
    NUM_K_IVDEP
    for (idx i = 0; i < n; ++i) {
        x[i] = value;
    }
}

/// @brief Copy a strided vector: \f$y \leftarrow x\f$.
template <typename T>
NUM_K_AINLINE void copy_strided(T *NUM_K_RESTRICT y, idx incy, const T *NUM_K_RESTRICT x, idx incx,
                                idx n) noexcept {
    NUM_K_IVDEP
    for (idx i = 0; i < n; ++i) {
        y[i * incy] = x[i * incx];
    }
}

/// @brief Copy and scale a strided vector: \f$y \leftarrow \alpha x\f$.
template <std::floating_point T>
NUM_K_AINLINE void scale_copy_strided(T *NUM_K_RESTRICT y, idx incy, const T *NUM_K_RESTRICT x,
                                      idx incx, T alpha, idx n) noexcept {
    NUM_K_IVDEP
    for (idx i = 0; i < n; ++i) {
        y[i * incy] = alpha * x[i * incx];
    }
}

/// @brief Add a multiple of one strided vector to another (AXPY): \f$y \leftarrow \alpha x + y\f$.
template <std::floating_point T>
NUM_K_AINLINE void axpy_strided(T *NUM_K_RESTRICT y, idx incy, const T *NUM_K_RESTRICT x, idx incx,
                                T alpha, idx n) noexcept {
    NUM_K_IVDEP
    for (idx i = 0; i < n; ++i) {
        y[i * incy] += alpha * x[i * incx];
    }
}

/// @brief Compute the squared Euclidean norm of a strided vector: \f$\|x\|_2^2\f$.
template <std::floating_point T>
NUM_K_AINLINE T norm_sq_strided(const T *NUM_K_RESTRICT x, idx incx, idx n) noexcept {
    return detail::reduce<T>(n, [x, incx](idx i) { return x[i * incx] * x[i * incx]; });
}

/// @brief Swap two strided vectors: \f$x \leftrightarrow y\f$.
template <typename T>
NUM_K_AINLINE void swap_strided(T *x, idx incx, T *y, idx incy, idx n) noexcept {
    for (idx i = 0; i < n; ++i)
        std::swap(x[i * incx], y[i * incy]);
}

/// @brief Scale a vector (SCAL): \f$x \leftarrow \alpha x\f$.
template <std::floating_point T>
NUM_K_AINLINE void scale(T *NUM_K_RESTRICT x, T alpha, idx n) noexcept {
    NUM_K_IVDEP
    for (idx i = 0; i < n; ++i) {
        x[i] *= alpha;
    }
}

/// @brief Add a multiple of one vector to another (AXPY): \f$y \leftarrow \alpha x + y\f$.
template <std::floating_point T>
NUM_K_AINLINE void axpy(T *NUM_K_RESTRICT y, const T *NUM_K_RESTRICT x, T alpha, idx n) noexcept {
    NUM_K_IVDEP
    for (idx i = 0; i < n; ++i) {
        y[i] += alpha * x[i];
    }
}

/// @brief Scale and add (AXPBY): \f$y \leftarrow a x + b y\f$.
template <std::floating_point T>
NUM_K_AINLINE void axpby(T *NUM_K_RESTRICT y, const T *NUM_K_RESTRICT x, T a, T b, idx n) noexcept {
    NUM_K_IVDEP
    for (idx i = 0; i < n; ++i) {
        y[i] = (a * x[i]) + (b * y[i]);
    }
}

/// @brief Form a linear combination of two vectors: \f$z \leftarrow a x + b y\f$.
template <std::floating_point T>
NUM_K_AINLINE void axpbyz(T *NUM_K_RESTRICT z, const T *NUM_K_RESTRICT x, const T *NUM_K_RESTRICT y,
                          T a, T b, idx n) noexcept {
    NUM_K_IVDEP
    for (idx i = 0; i < n; ++i) {
        z[i] = (a * x[i]) + (b * y[i]);
    }
}

/// @brief Compute the dot product: \f$x^T y\f$.
/// @note Sums are reordered for speed; the `contract::ordered` overload is reproducible.
template <std::floating_point T>
[[nodiscard]] NUM_K_AINLINE T dot(const T *NUM_K_RESTRICT x, const T *NUM_K_RESTRICT y,
                                  idx n) noexcept {
    return detail::reduce<T>(n, [x, y](idx i) { return x[i] * y[i]; });
}

/// @brief Compute the dot product, summing in index order: \f$x^T y\f$.
template <std::floating_point T>
[[nodiscard]] NUM_K_AINLINE T dot(contract::ordered_t, const T *NUM_K_RESTRICT x,
                                  const T *NUM_K_RESTRICT y, idx n) noexcept {
    T s = T(0);
    NUM_K_IVDEP
    for (idx i = 0; i < n; ++i) {
        s += x[i] * y[i];
    }
    return s;
}

/// @brief Compute the dot product: \f$x^T y\f$; the same as the untagged overload.
template <std::floating_point T>
[[nodiscard]] NUM_K_AINLINE T dot(contract::throughput_t, const T *NUM_K_RESTRICT x,
                                  const T *NUM_K_RESTRICT y, idx n) noexcept {
    return dot(x, y, n);
}

/// @brief Result of `dot2`: \f$x^T y\f$ and \f$x^T z\f$.
template <std::floating_point T>
struct dot2_result {
    T xy;
    T xz;
};

/// @brief Compute two dot products in one pass: \f$(x^T y,\ x^T z)\f$.
template <std::floating_point T>
[[nodiscard]] NUM_K_AINLINE dot2_result<T> dot2(const T *NUM_K_RESTRICT x,
                                                const T *NUM_K_RESTRICT y,
                                                const T *NUM_K_RESTRICT z, idx n) noexcept {
    // Half the chains per result: two reductions share the registers.
    constexpr idx lanes = detail::reduction_lanes / 2;
    T xy[lanes]{};
    T xz[lanes]{};
    idx i = 0;
    for (; i + lanes <= n; i += lanes) {
        NUM_K_IVDEP
        for (idx k = 0; k < lanes; ++k) {
            const T xi = x[i + k];
            xy[k] += xi * y[i + k];
            xz[k] += xi * z[i + k];
        }
    }
    for (idx width = lanes / 2; width > 0; width /= 2) {
        for (idx k = 0; k < width; ++k) {
            xy[k] += xy[k + width];
            xz[k] += xz[k + width];
        }
    }
    T sum_xy = xy[0];
    T sum_xz = xz[0];
    for (; i < n; ++i) {
        const T xi = x[i];
        sum_xy += xi * y[i];
        sum_xz += xi * z[i];
    }
    return {sum_xy, sum_xz};
}

/// @brief Result of `dot_norm_sq`: \f$x^T y\f$ and \f$\|y\|_2^2\f$.
template <std::floating_point T>
struct dot_norm_result {
    T dot;
    T norm_sq;
};

/// @brief Compute a dot product and a squared norm in one pass: \f$(x^T y,\ \|y\|_2^2)\f$.
template <std::floating_point T>
[[nodiscard]] NUM_K_AINLINE dot_norm_result<T>
dot_norm_sq(const T *NUM_K_RESTRICT x, const T *NUM_K_RESTRICT y, idx n) noexcept {
    constexpr idx lanes = detail::reduction_lanes / 2;
    T xy[lanes]{};
    T yy[lanes]{};
    idx i = 0;
    for (; i + lanes <= n; i += lanes) {
        NUM_K_IVDEP
        for (idx k = 0; k < lanes; ++k) {
            const T yi = y[i + k];
            xy[k] += x[i + k] * yi;
            yy[k] += yi * yi;
        }
    }
    for (idx width = lanes / 2; width > 0; width /= 2) {
        for (idx k = 0; k < width; ++k) {
            xy[k] += xy[k + width];
            yy[k] += yy[k + width];
        }
    }
    T sum_xy = xy[0];
    T sum_yy = yy[0];
    for (; i < n; ++i) {
        const T yi = y[i];
        sum_xy += x[i] * yi;
        sum_yy += yi * yi;
    }
    return {sum_xy, sum_yy};
}

/// @brief Add a multiple of one vector to another and return the new squared norm: \f$\|y\|_2^2\f$.
template <std::floating_point T>
[[nodiscard]] NUM_K_AINLINE T axpy_norm_sq(T *NUM_K_RESTRICT y, const T *NUM_K_RESTRICT x, T alpha,
                                           idx n) noexcept {
    return detail::reduce<T>(n, [y, x, alpha](idx i) {
        const T yi = y[i] + (alpha * x[i]);
        y[i] = yi;
        return yi * yi;
    });
}

/// @brief Compute the squared norm of a linear combination without forming it:
/// \f$\|a x + b y\|_2^2\f$.
template <std::floating_point T>
[[nodiscard]] NUM_K_AINLINE T linear_combination_norm_sq(const T *NUM_K_RESTRICT x, T a,
                                                         const T *NUM_K_RESTRICT y, T b,
                                                         idx n) noexcept {
    return detail::reduce<T>(n, [x, a, y, b](idx i) {
        const T value = (a * x[i]) + (b * y[i]);
        return value * value;
    });
}

/// @brief Compute the squared Euclidean norm: \f$\|x\|_2^2\f$.
template <std::floating_point T>
[[nodiscard]] NUM_K_AINLINE T norm_sq(const T *NUM_K_RESTRICT x, idx n) noexcept {
    return detail::reduce<T>(n, [x](idx i) { return x[i] * x[i]; });
}

// Declared early for norm's rescaled path; ADL cannot find it later.
template <std::floating_point T>
[[nodiscard]] NUM_K_AINLINE T linf_norm(const T *NUM_K_RESTRICT x, idx n) noexcept;

/// @brief Compute the Euclidean norm: \f$\|x\|_2\f$.
/// Rescales to avoid overflow and underflow.
template <std::floating_point T>
[[nodiscard]] NUM_K_AINLINE T norm(const T *NUM_K_RESTRICT x, idx n) noexcept {
    const T squared = norm_sq(x, n);
    if (squared > T(0) && std::isfinite(squared)) {
        return std::sqrt(squared);
    }
    if (std::isnan(squared)) {
        return squared;
    }
    // Overflow or underflow: factor out the largest magnitude before squaring.
    const T scale = linf_norm(x, n);
    if (scale == T(0) || !std::isfinite(scale)) {
        return scale;
    }
    const T scaled = detail::reduce<T>(n, [x, scale](idx i) {
        const T v = x[i] / scale;
        return v * v;
    });
    return scale * std::sqrt(scaled);
}

/// @brief Compute the 1-norm: \f$\|x\|_1 = \sum_i |x_i|\f$.
template <std::floating_point T>
[[nodiscard]] NUM_K_AINLINE T l1_norm(const T *NUM_K_RESTRICT x, idx n) noexcept {
    return detail::reduce<T>(n, [x](idx i) { return std::abs(x[i]); });
}

/// @brief Compute the infinity norm: \f$\|x\|_\infty = \max_i |x_i|\f$.
template <std::floating_point T>
[[nodiscard]] NUM_K_AINLINE T linf_norm(const T *NUM_K_RESTRICT x, idx n) noexcept {
    T mx = T(0);
    for (idx i = 0; i < n; ++i) {
        const T v = std::abs(x[i]);
        if (v > mx) {
            mx = v;
        }
    }
    return mx;
}

/// @brief Sum the entries of a vector: \f$\sum_i x_i\f$.
template <std::floating_point T>
[[nodiscard]] NUM_K_AINLINE T sum(const T *NUM_K_RESTRICT x, idx n) noexcept {
    return detail::reduce<T>(n, [x](idx i) { return x[i]; });
}

/// @brief Clamp each entry to an interval:
/// \f$x_i \leftarrow \min(\max(x_i, \mathrm{lo}), \mathrm{hi})\f$.
template <std::floating_point T>
NUM_K_AINLINE void clamp(T *NUM_K_RESTRICT x, T lo, T hi, idx n) noexcept {
    NUM_K_IVDEP
    for (idx i = 0; i < n; ++i) {
        x[i] = std::clamp(x[i], lo, hi);
    }
}

/// @brief Add two vectors: \f$z \leftarrow x + y\f$.
template <std::floating_point T>
NUM_K_AINLINE void add(T *NUM_K_RESTRICT z, const T *NUM_K_RESTRICT x, const T *NUM_K_RESTRICT y,
                       idx n) noexcept {
    NUM_K_IVDEP
    for (idx i = 0; i < n; ++i) {
        z[i] = x[i] + y[i];
    }
}

/// @brief Multiply two vectors entry by entry: \f$z_i \leftarrow x_i y_i\f$.
template <std::floating_point T>
NUM_K_AINLINE void hadamard_mul(T *NUM_K_RESTRICT z, const T *NUM_K_RESTRICT x,
                                const T *NUM_K_RESTRICT y, idx n) noexcept {
    NUM_K_IVDEP
    for (idx i = 0; i < n; ++i) {
        z[i] = x[i] * y[i];
    }
}

/// @brief Divide two vectors entry by entry: \f$z_i \leftarrow x_i / y_i\f$.
template <std::floating_point T>
NUM_K_AINLINE void hadamard_div(T *NUM_K_RESTRICT z, const T *NUM_K_RESTRICT x,
                                const T *NUM_K_RESTRICT y, idx n) noexcept {
    NUM_K_IVDEP
    for (idx i = 0; i < n; ++i) {
        z[i] = x[i] / y[i];
    }
}

/// @brief Take the reciprocal of each entry: \f$y_i \leftarrow 1 / x_i\f$.
template <std::floating_point T>
NUM_K_AINLINE void inv(T *NUM_K_RESTRICT y, const T *NUM_K_RESTRICT x, idx n) noexcept {
    NUM_K_IVDEP
    for (idx i = 0; i < n; ++i) {
        y[i] = T(1) / x[i];
    }
}

/// @brief Swap two vectors: \f$x \leftrightarrow y\f$.
template <typename T>
NUM_K_AINLINE void swap(T *NUM_K_RESTRICT x, T *NUM_K_RESTRICT y, idx n) noexcept {
    NUM_K_IVDEP
    for (idx i = 0; i < n; ++i) {
        std::swap(x[i], y[i]);
    }
}

/// @brief Find the entry of largest magnitude (IAMAX): \f$\arg\max_i |x_i|\f$.
/// Returns 0 when `n` is 0.
template <std::floating_point T>
[[nodiscard]] NUM_K_AINLINE idx argmax_abs(const T *NUM_K_RESTRICT x, idx n) noexcept {
    if (n == 0) {
        return 0;
    }
    idx best_idx = 0;
    T best_val = std::abs(x[0]);
    for (idx i = 1; i < n; ++i) {
        const T val = std::abs(x[i]);
        if (val > best_val) {
            best_val = val;
            best_idx = i;
        }
    }
    return best_idx;
}

/// @brief Construct a Givens rotation (ROTG):
/// \f$c, s\f$ with \f$(c a + s b,\ c b - s a) = (r, 0)\f$.
template <std::floating_point T>
NUM_K_AINLINE void rotg(T a, T b, T &c, T &s) noexcept {
    if (b == T(0)) {
        c = T(1);
        s = T(0);
    } else if (a == T(0)) {
        c = T(0);
        s = T(1);
    } else {
        T r = std::hypot(a, b);
        c = a / r;
        s = b / r;
    }
}

/// @brief Apply a Givens rotation (ROT):
/// \f$(x_i, y_i) \leftarrow (c x_i + s y_i,\ c y_i - s x_i)\f$.
template <std::floating_point T>
NUM_K_AINLINE void rot(T *NUM_K_RESTRICT x, T *NUM_K_RESTRICT y, T c, T s, idx n) noexcept {
    NUM_K_IVDEP
    for (idx i = 0; i < n; ++i) {
        T xi = x[i];
        T yi = y[i];
        x[i] = (c * xi) + (s * yi);
        y[i] = (-s * xi) + (c * yi);
    }
}

} // namespace num::kernel
