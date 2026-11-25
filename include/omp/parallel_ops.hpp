/// @file omp/parallel_ops.hpp
/// @brief Threaded blocked loops and reductions over the raw kernels.
/// Results do not depend on the thread count.
#pragma once

#include "core/types.hpp"
#include <algorithm>

namespace num::omp {

/// @brief Elements per thread block.
inline constexpr idx parallel_block = idx{1} << 14;

/// @brief Default single-thread threshold. Override with `-DNUMERICS_PARALLEL_THRESHOLD=<n>`.
#ifndef NUMERICS_PARALLEL_THRESHOLD
#define NUMERICS_PARALLEL_THRESHOLD (1 << 18)
#endif
/// @brief Element count below which an operation stays on one thread.
/// @note Starting a parallel region costs about 34 µs on macOS libomp.
inline constexpr idx parallel_threshold = idx{NUMERICS_PARALLEL_THRESHOLD};

/// @brief Maximum number of blocks.
inline constexpr idx max_parallel_blocks = 256;

/// @brief Block size for `n` elements: at least `parallel_block`, at most `max_parallel_blocks`.
[[nodiscard]] inline constexpr idx block_size_for(idx n) noexcept {
    const idx even_split = (n + max_parallel_blocks - 1) / max_parallel_blocks;
    return even_split > parallel_block ? even_split : parallel_block;
}

/// @brief Number of blocks covering `n` elements.
[[nodiscard]] inline constexpr idx block_count_for(idx n) noexcept {
    const idx size = block_size_for(n);
    return (n + size - 1) / size;
}

/// @brief Blocked reduction: \f$\sum_b \mathrm{block}(o_b, \ell_b)\f$ over \f$[0, n)\f$.
/// Partials are summed in block order.
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

/// @brief Blocked loop: `block(offset, length)` over \f$[0, n)\f$.
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
