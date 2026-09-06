/// @file kernel/vector.hpp
/// @brief Raw-pointer kernels: BLAS-1 vector ops and fused reductions.
///
/// SPDX-License-Identifier: MIT
/// Part of numerics, (c) 2026 Aditya Dendukuri.
/// https://github.com/AdityaDendukuri/numerics
///
/// This file has no dependencies outside the standard library: copy it into
/// another project as-is, or lift a single routine out of it together with the
/// NUM_K_* macro block below. Please keep the two attribution lines above with
/// whatever you take.
///
/// Kernels assume non-owning, caller-sized buffers and do not allocate. This is
/// the base file of the `num::kernel` tier: `dense.hpp`, `sparse.hpp`, and
/// `rotations.hpp` all build on the macros, the `contract` tags, and
/// `detail::reduce` defined here.
///
/// @section kernel_contract The contract every kernel in this tier assumes
///
/// These rules hold for **every** function in `num::kernel` and are not repeated
/// on each one. A per-function `@pre` states only what is additional to this.
///
/// - **Nothing is checked.** No dimension is validated, no pointer is tested for
///   null, no divisor is tested for zero. Violating a precondition is undefined
///   behaviour, not a thrown exception. Proving the preconditions is the job of
///   the typed layers above (`num::vec`, `num::mat`, the `algebra`
///   concepts); this tier assumes the proof already happened.
/// - **Every buffer is caller-allocated and caller-sized.** A kernel never
///   allocates, never frees, never grows a buffer, and never keeps a pointer past
///   the call. A parameter documented as length `n` must be readable (and, if an
///   output, writable) for exactly `n` elements.
/// - **Pointers marked `NUM_K_RESTRICT` must not overlap.** That is nearly all of
///   them, and it is load-bearing: it is what lets the compiler vectorize these
///   loops. Passing the same buffer as two restrict-qualified parameters is
///   undefined behaviour and will silently produce wrong results at `-O2`, not a
///   diagnostic. Where in-place operation *is* permitted (`x` and `y` may be the
///   same pointer), the function says so explicitly under `@aliasing`.
/// - **`noexcept` throughout.** A kernel has no failure mode it can report by
///   throwing. Routines that can fail numerically return `bool` or a result
///   struct instead.
/// - **Reductions are not in source order** unless the name or a
///   `contract::ordered` overload says they are; see `detail::reduce` for the
///   summation-order guarantee and its limits.
///
/// Complexity is quoted in elements touched, not in FLOPs, since every routine
/// here is bandwidth-bound at realistic sizes.
#pragma once

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstddef>

namespace num {
/// Index type. Identical to the definition in container/types.hpp, and repeating
/// an identical typedef is well-formed, so this file stays self-contained whether
/// or not the rest of numerics is present.
using idx = std::size_t;
} // namespace num

#if defined(__GNUC__) || defined(__clang__)
#define NUM_K_AINLINE [[gnu::always_inline]] inline
#define NUM_K_RESTRICT __restrict__
#define NUM_K_IVDEP _Pragma("GCC ivdep")
// GCC/clang vector extension. Used only to give the reductions an accumulator
// as wide as the target's registers; everything still compiles without it.
#define NUM_K_VECTOR_EXT 1
#else
#define NUM_K_AINLINE inline
#define NUM_K_RESTRICT
#define NUM_K_IVDEP
#endif

// Bytes in the widest vector register the target actually has. Only consulted
// on the NUM_K_VECTOR_EXT path.
#if defined(__AVX512F__)
#define NUM_K_VECTOR_BYTES 64
#elif defined(__AVX__)
#define NUM_K_VECTOR_BYTES 32
#else
#define NUM_K_VECTOR_BYTES 16
#endif

namespace num::kernel {

/// Computational contracts used at the unchecked kernel boundary.  The typed
/// layers are responsible for proving these preconditions before selecting a
/// raw overload; the tags make an intentional relaxation visible at the call
/// site without adding runtime state.
namespace contract {
struct alias_safe_t final {};
struct throughput_t final {};
struct ordered_t final {};

inline constexpr alias_safe_t alias_safe{};
inline constexpr throughput_t throughput{};
inline constexpr ordered_t ordered{};
} // namespace contract

namespace detail {

/// @brief Independent accumulation chains used by every reduction here.
///
/// A reduction written as one accumulator carries a loop-carried floating-point
/// dependency. Because addition is not associative the compiler may not break
/// it, so the loop runs at the latency of the adder rather than its throughput —
/// measured at three to four times slower here, and the gap persists past cache
/// size because the loop is latency bound, not bandwidth bound.
inline constexpr idx reduction_lanes = 8;

#if defined(NUM_K_VECTOR_EXT)

/// @brief Independent *vector* accumulators on the vector-extension path.
inline constexpr idx reduction_accumulators = 4;

/// @brief A native vector of `Bytes` bytes holding elements of type T.
///
/// An array of scalar lanes is not enough on its own: clang packs such an array
/// into 128-bit registers and never widens it, whatever the lane count, so on
/// AVX2 it would use half the register width and on AVX-512 an eighth. Naming
/// the vector type explicitly is what makes the accumulator as wide as the
/// hardware.
template <class T, idx Bytes>
struct vector_of {
    typedef T type __attribute__((vector_size(Bytes)));
};

#endif

/// @brief Reduce `element(i)` over `[0, n)` with several accumulators in flight.
///
/// Each index is visited exactly once, so `element` may carry a side effect —
/// that is how the fused update-and-reduce kernels work.
///
/// ### Summation order
///
/// The visiting order is a permutation of the source order: indices are spread
/// across accumulators and combined pairwise at the end. Bounding the chain
/// length this way makes the error grow like \f$O(n/K)\f$ instead of
/// \f$O(n)\f$, so on unstructured data this is more accurate than source order
/// as well as faster. It is *less* accurate on data whose sign pattern is
/// periodic in the accumulator count, where source order cancels adjacent terms
/// immediately.
///
/// The grouping is fixed for a given build but **depends on the target's vector
/// width**, so a result computed under AVX-512 need not match the same source
/// built for SSE bit for bit. Callers who need a result reproducible across
/// machines, or exact source-order semantics, select `contract::ordered`.
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
    // One vector at a time, so the scalar tail is shorter than a vector rather
    // than shorter than a whole block. Without this a vector of 20 elements
    // would be reduced entirely serially on a 512-bit target.
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
    // Portable fallback: independent scalar chains. Breaks the dependency, but
    // leaves the compiler to choose the vector width on its own.
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

// vec Level-1 BLAS & Fused Kernels

/// @brief Copies a vector, \f$y_i \leftarrow x_i\f$.
template <typename T>
NUM_K_AINLINE void copy(T *NUM_K_RESTRICT y, const T *NUM_K_RESTRICT x, idx n) noexcept {
    NUM_K_IVDEP
    for (idx i = 0; i < n; ++i) {
        y[i] = x[i];
    }
}

/// @brief Fills a vector, \f$x_i \leftarrow value\f$.
template <typename T>
NUM_K_AINLINE void fill(T *NUM_K_RESTRICT x, T value, idx n) noexcept {
    NUM_K_IVDEP
    for (idx i = 0; i < n; ++i) {
        x[i] = value;
    }
}

/// @brief Copies strided vectors, \f$y_i \leftarrow x_i\f$.
template <typename T>
NUM_K_AINLINE void copy_strided(T *NUM_K_RESTRICT y, idx incy, const T *NUM_K_RESTRICT x, idx incx,
                                idx n) noexcept {
    NUM_K_IVDEP
    for (idx i = 0; i < n; ++i) {
        y[i * incy] = x[i * incx];
    }
}

/// @brief Copies and scales strided vectors, \f$y_i \leftarrow \alpha x_i\f$.
template <std::floating_point T>
NUM_K_AINLINE void scale_copy_strided(T *NUM_K_RESTRICT y, idx incy, const T *NUM_K_RESTRICT x,
                                      idx incx, T alpha, idx n) noexcept {
    NUM_K_IVDEP
    for (idx i = 0; i < n; ++i) {
        y[i * incy] = alpha * x[i * incx];
    }
}

/// @brief Strided vector update, \f$y_i \leftarrow y_i + \alpha x_i\f$.
template <std::floating_point T>
NUM_K_AINLINE void axpy_strided(T *NUM_K_RESTRICT y, idx incy, const T *NUM_K_RESTRICT x, idx incx,
                                T alpha, idx n) noexcept {
    NUM_K_IVDEP
    for (idx i = 0; i < n; ++i) {
        y[i * incy] += alpha * x[i * incx];
    }
}

/// @brief Squared Euclidean norm over a strided vector.
template <std::floating_point T>
NUM_K_AINLINE T norm_sq_strided(const T *NUM_K_RESTRICT x, idx incx, idx n) noexcept {
    return detail::reduce<T>(n, [x, incx](idx i) { return x[i * incx] * x[i * incx]; });
}

/// @brief Swaps two strided vectors elementwise.
template <typename T>
NUM_K_AINLINE void swap_strided(T *x, idx incx, T *y, idx incy, idx n) noexcept {
    for (idx i = 0; i < n; ++i)
        std::swap(x[i * incx], y[i * incy]);
}

/// @brief Computes in-place scalar scaling \f$x_i \leftarrow \alpha x_i\f$.
template <std::floating_point T>
NUM_K_AINLINE void scale(T *NUM_K_RESTRICT x, T alpha, idx n) noexcept {
    NUM_K_IVDEP
    for (idx i = 0; i < n; ++i) {
        x[i] *= alpha;
    }
}

/// @brief Computes vector update \f$y_i \leftarrow y_i + \alpha x_i\f$ (BLAS AXPY).
template <std::floating_point T>
NUM_K_AINLINE void axpy(T *NUM_K_RESTRICT y, const T *NUM_K_RESTRICT x, T alpha, idx n) noexcept {
    NUM_K_IVDEP
    for (idx i = 0; i < n; ++i) {
        y[i] += alpha * x[i];
    }
}

/// @brief Computes fused scaled vector update \f$y_i \leftarrow a x_i + b y_i\f$ in a single memory
/// pass.
template <std::floating_point T>
NUM_K_AINLINE void axpby(T *NUM_K_RESTRICT y, const T *NUM_K_RESTRICT x, T a, T b, idx n) noexcept {
    NUM_K_IVDEP
    for (idx i = 0; i < n; ++i) {
        y[i] = (a * x[i]) + (b * y[i]);
    }
}

/// @brief Computes vector linear combination \f$z_i \leftarrow a x_i + b y_i\f$.
template <std::floating_point T>
NUM_K_AINLINE void axpbyz(T *NUM_K_RESTRICT z, const T *NUM_K_RESTRICT x, const T *NUM_K_RESTRICT y,
                          T a, T b, idx n) noexcept {
    NUM_K_IVDEP
    for (idx i = 0; i < n; ++i) {
        z[i] = (a * x[i]) + (b * y[i]);
    }
}

/// @brief Computes vector dot product \f$\mathbf{x} \cdot \mathbf{y} = \sum_{i=0}^{n-1} x_i y_i\f$.
///
/// Accumulates in `detail::reduction_lanes` independent chains. See that
/// constant for why this is both faster and, on unstructured data, more
/// accurate than source order — and for the one data shape where it is not.
/// Select `contract::ordered` when exact source-order summation is required.
template <std::floating_point T>
[[nodiscard]] NUM_K_AINLINE T dot(const T *NUM_K_RESTRICT x, const T *NUM_K_RESTRICT y,
                                  idx n) noexcept {
    return detail::reduce<T>(n, [x, y](idx i) { return x[i] * y[i]; });
}

/// @brief Source-ordered dot product: sums strictly in increasing index order.
///
/// Slower, and on unstructured data less accurate. Use it when a result must
/// reproduce a reference implementation bit for bit, or when the data's sign
/// pattern is periodic in `detail::reduction_lanes`.
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

/// @brief Retained spelling for callers that opted into throughput explicitly.
///
/// Now the same kernel as the unqualified overload, which no longer has to be
/// asked for it.
template <std::floating_point T>
[[nodiscard]] NUM_K_AINLINE T dot(contract::throughput_t, const T *NUM_K_RESTRICT x,
                                  const T *NUM_K_RESTRICT y, idx n) noexcept {
    return dot(x, y, n);
}

/// @brief Computes two inner products sharing the same left operand in one pass.
///
/// This is the primitive needed by fused Krylov recurrences: `x` is loaded once
/// instead of once per reduction.  The result members are deliberately named so
/// their mathematical meaning remains clear at call sites.
template <std::floating_point T>
struct dot2_result {
    T xy;
    T xz;
};

template <std::floating_point T>
[[nodiscard]] NUM_K_AINLINE dot2_result<T> dot2(const T *NUM_K_RESTRICT x,
                                                const T *NUM_K_RESTRICT y,
                                                const T *NUM_K_RESTRICT z, idx n) noexcept {
    // Half the usual chain count per result: two reductions share the register
    // budget, and the shared load of x already supplies part of the parallelism.
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

/// @brief Computes an inner product and squared norm in one memory pass.
template <std::floating_point T>
struct dot_norm_result {
    T dot;
    T norm_sq;
};

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

/// @brief Updates `y <- y + alpha*x` and returns the new squared norm of `y`.
///
/// The update and convergence statistic share one traversal.  This is generally
/// bandwidth-optimal for iterative methods because the updated values remain in
/// registers for the reduction.
template <std::floating_point T>
[[nodiscard]] NUM_K_AINLINE T axpy_norm_sq(T *NUM_K_RESTRICT y, const T *NUM_K_RESTRICT x, T alpha,
                                           idx n) noexcept {
    return detail::reduce<T>(n, [y, x, alpha](idx i) {
        const T yi = y[i] + (alpha * x[i]);
        y[i] = yi;
        return yi * yi;
    });
}

/// @brief Returns \f$\|a x + b y\|_2^2\f$ without materializing the combination.
template <std::floating_point T>
[[nodiscard]] NUM_K_AINLINE T linear_combination_norm_sq(const T *NUM_K_RESTRICT x, T a,
                                                         const T *NUM_K_RESTRICT y, T b,
                                                         idx n) noexcept {
    return detail::reduce<T>(n, [x, a, y, b](idx i) {
        const T value = (a * x[i]) + (b * y[i]);
        return value * value;
    });
}

/// @brief Computes squared Euclidean norm \f$\|\mathbf{x}\|_2^2 = \sum_{i=0}^{n-1} x_i^2\f$.
template <std::floating_point T>
[[nodiscard]] NUM_K_AINLINE T norm_sq(const T *NUM_K_RESTRICT x, idx n) noexcept {
    return detail::reduce<T>(n, [x](idx i) { return x[i] * x[i]; });
}

// Declared here so the rescaled path in `norm` can reach it; a pointer to a
// fundamental type has no associated namespace, so ADL cannot find it later.
template <std::floating_point T>
[[nodiscard]] NUM_K_AINLINE T linf_norm(const T *NUM_K_RESTRICT x, idx n) noexcept;

/// @brief Computes Euclidean \f$L_2\f$ norm \f$\|\mathbf{x}\|_2 = \sqrt{\sum_{i=0}^{n-1} x_i^2}\f$.
///
/// The direct form `sqrt(norm_sq(x, n))` squares before it sums, so it overflows
/// to infinity once \f$\|x\|_2\f$ passes about `1.3e154` in double precision,
/// and flushes to zero below about `1.5e-154` — in both cases returning a wrong
/// answer for a vector every element of which is a perfectly ordinary finite
/// number. The direct form is taken whenever its result is usable, and only a
/// non-finite or zero sum pays for the rescaled second pass.
template <std::floating_point T>
[[nodiscard]] NUM_K_AINLINE T norm(const T *NUM_K_RESTRICT x, idx n) noexcept {
    const T squared = norm_sq(x, n);
    if (squared > T(0) && std::isfinite(squared)) {
        return std::sqrt(squared);
    }
    if (std::isnan(squared)) {
        return squared;
    }
    // Either the sum overflowed, or every term underflowed to zero. Both are
    // fixed by factoring out the largest magnitude before squaring.
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

/// @brief Computes \f$L_1\f$ norm \f$\|\mathbf{x}\|_1 = \sum_{i=0}^{n-1} |x_i|\f$.
template <std::floating_point T>
[[nodiscard]] NUM_K_AINLINE T l1_norm(const T *NUM_K_RESTRICT x, idx n) noexcept {
    return detail::reduce<T>(n, [x](idx i) { return std::abs(x[i]); });
}

/// @brief Computes \f$L_\infty\f$ norm \f$\|\mathbf{x}\|_\infty = \max_{0 \le i < n} |x_i|\f$.
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

/// @brief Computes scalar vector sum \f$\sum_{i=0}^{n-1} x_i\f$.
template <std::floating_point T>
[[nodiscard]] NUM_K_AINLINE T sum(const T *NUM_K_RESTRICT x, idx n) noexcept {
    return detail::reduce<T>(n, [x](idx i) { return x[i]; });
}

/// @brief In-place vector element clamp \f$x_i \leftarrow \min(\max(x_i, x_{\min}), x_{\max})\f$.
template <std::floating_point T>
NUM_K_AINLINE void clamp(T *NUM_K_RESTRICT x, T lo, T hi, idx n) noexcept {
    NUM_K_IVDEP
    for (idx i = 0; i < n; ++i) {
        x[i] = std::clamp(x[i], lo, hi);
    }
}

/// @brief Computes elementwise vector sum \f$z_i = x_i + y_i\f$.
template <std::floating_point T>
NUM_K_AINLINE void add(T *NUM_K_RESTRICT z, const T *NUM_K_RESTRICT x, const T *NUM_K_RESTRICT y,
                       idx n) noexcept {
    NUM_K_IVDEP
    for (idx i = 0; i < n; ++i) {
        z[i] = x[i] + y[i];
    }
}

/// @brief Computes elementwise Hadamard product \f$z_i = x_i \cdot y_i\f$.
template <std::floating_point T>
NUM_K_AINLINE void hadamard_mul(T *NUM_K_RESTRICT z, const T *NUM_K_RESTRICT x,
                                const T *NUM_K_RESTRICT y, idx n) noexcept {
    NUM_K_IVDEP
    for (idx i = 0; i < n; ++i) {
        z[i] = x[i] * y[i];
    }
}

/// @brief Computes elementwise division \f$z_i = x_i / y_i\f$.
template <std::floating_point T>
NUM_K_AINLINE void hadamard_div(T *NUM_K_RESTRICT z, const T *NUM_K_RESTRICT x,
                                const T *NUM_K_RESTRICT y, idx n) noexcept {
    NUM_K_IVDEP
    for (idx i = 0; i < n; ++i) {
        z[i] = x[i] / y[i];
    }
}

/// @brief Computes elementwise reciprocal \f$y_i = 1 / x_i\f$ (e.g. for Jacobi preconditioners).
template <std::floating_point T>
NUM_K_AINLINE void inv(T *NUM_K_RESTRICT y, const T *NUM_K_RESTRICT x, idx n) noexcept {
    NUM_K_IVDEP
    for (idx i = 0; i < n; ++i) {
        y[i] = T(1) / x[i];
    }
}

// Row / Column Swaps & Search

/// @brief Swaps vector buffers \f$\mathbf{x} \leftrightarrow \mathbf{y}\f$ of length \f$n\f$
/// in-place.
template <typename T>
NUM_K_AINLINE void swap(T *NUM_K_RESTRICT x, T *NUM_K_RESTRICT y, idx n) noexcept {
    NUM_K_IVDEP
    for (idx i = 0; i < n; ++i) {
        std::swap(x[i], y[i]);
    }
}

/// @brief Returns index of element with maximum absolute value \f$\arg\max_i |x_i|\f$ in vector
/// \f$\mathbf{x}\f$.
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

} // namespace num::kernel
