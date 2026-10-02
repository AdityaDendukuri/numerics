/// @file omp/parallel_ops.hpp
/// @brief Threaded block decomposition and reduction over the raw kernels.
///
/// Each thread calls the raw kernel on a block, and block partials are summed in index order,
/// so the result does not depend on the thread count. See the backends page.
///
/// Independent of `vec<real>`/`mat<real>`, so container headers can use it without a circular include.
#pragma once

#include "core/types.hpp"
#include <algorithm>

namespace num::omp {

/// @brief Elements handled by one thread's call into a raw kernel.
inline constexpr idx parallel_block = idx{1} << 14;

/// @brief Below this element count an operation stays on one thread. Starting a parallel region
/// costs about 34 us on macOS libomp. Override with `-DNUMERICS_PARALLEL_THRESHOLD=<n>`.
#ifndef NUMERICS_PARALLEL_THRESHOLD
#define NUMERICS_PARALLEL_THRESHOLD (1 << 18)
#endif
/// @brief The element count below which an operation stays on one thread.
inline constexpr idx parallel_threshold = idx{NUMERICS_PARALLEL_THRESHOLD};

/// @brief Upper bound on blocks, so the partial-sum buffer can live on the stack.
inline constexpr idx max_parallel_blocks = 256;

/// @brief Element count per block for a problem of size `n`.
///
/// At least `parallel_block`, and large enough that no more than
/// `max_parallel_blocks` blocks are produced.
[[nodiscard]] inline constexpr idx block_size_for(idx n) noexcept {
    const idx even_split = (n + max_parallel_blocks - 1) / max_parallel_blocks;
    return even_split > parallel_block ? even_split : parallel_block;
}

/// @brief Number of blocks covering `n` elements.
[[nodiscard]] inline constexpr idx block_count_for(idx n) noexcept {
    const idx size = block_size_for(n);
    return (n + size - 1) / size;
}

/// @brief Sum `block(offset, length)` over a blocked decomposition of `[0, n)`.
///
/// `block` is expected to be a raw kernel call over the slice, returning its
/// partial. Runs on one thread below `parallel_threshold`, or when the build has
/// no OpenMP. The summation order of the partials is fixed regardless.
template <class T, class Block>
[[nodiscard]] inline T parallel_reduce(idx n, Block block) {
    if (n == 0) {
        return T(0);
    }
#if defined(NUMERICS_HAS_OMP)
    if (n >= parallel_threshold) {
        const idx size = block_size_for(n);
        const idx blocks = (n + size - 1) / size;
        T partial[max_parallel_blocks]{};
#pragma omp parallel for schedule(static)
        for (idx b = 0; b < blocks; ++b) {
            const idx offset = b * size;
            partial[b] = block(offset, std::min(size, n - offset));
        }
        // Fixed order, so the result does not vary with the thread count.
        T total = T(0);
        for (idx b = 0; b < blocks; ++b) {
            total += partial[b];
        }
        return total;
    }
#endif
    return block(idx{0}, n);
}

/// @brief Apply `block(offset, length)` over a blocked decomposition of `[0, n)`.
///
/// For elementwise work, where blocks are independent and no combination step is
/// needed. Deterministic by construction.
template <class Block>
inline void parallel_apply(idx n, Block block) {
    if (n == 0) {
        return;
    }
#if defined(NUMERICS_HAS_OMP)
    if (n >= parallel_threshold) {
        const idx size = block_size_for(n);
        const idx blocks = (n + size - 1) / size;
#pragma omp parallel for schedule(static)
        for (idx b = 0; b < blocks; ++b) {
            const idx offset = b * size;
            block(offset, std::min(size, n - offset));
        }
        return;
    }
#endif
    block(idx{0}, n);
}

} // namespace num::omp
