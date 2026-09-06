/// @file omp/parallel_ops.hpp
/// @brief Threaded block decomposition and reduction over the raw kernels.
///
/// Two rules shape everything here.
///
/// **A thread must not be handed a scalar loop.** Splitting a reduction across
/// threads and giving each one a single accumulator reintroduces exactly the
/// loop-carried dependency the raw kernels exist to break: every core then runs
/// at the latency of the adder, and eight of them reach roughly what one core
/// reaches running at throughput. Threads take a block each and call the raw
/// kernel on it, so the two levels of parallelism compose instead of cancelling.
///
/// **The answer must not depend on how many threads happened to run.** An
/// OpenMP `reduction(+:s)` clause combines the per-thread partials in an
/// unspecified order, so the same binary would return different bits under
/// `OMP_NUM_THREADS=4` and `=8`. Instead each block writes its partial to a
/// slot, and the slots are summed in index order afterwards. The block
/// decomposition is a function of `n` alone, so the result depends only on the
/// input and the build — never on the scheduler.
///
/// Independent of `vec`/`mat`, unlike the rest of `num::omp`: kept this
/// way so container-tier headers can use the block helpers without a circular
/// include back through `omp/vector_ops.hpp`.
#pragma once

#include "core/types.hpp"
#include <algorithm>

namespace num::omp {

/// @brief Elements handled by one thread's call into a raw kernel.
inline constexpr idx parallel_block = idx{1} << 14;

/// @brief Below this element count an operation stays on one thread.
///
/// Entering an OpenMP parallel region is not free: measured at roughly 34 us on
/// macOS libomp with eight threads, against about 6 us of actual work for a
/// 32k-element dot product. Threading below the crossover is not a small loss,
/// it is a large one — an order of magnitude at 16k elements — so the default
/// errs high. Tunable at configure time with `-DNUMERICS_PARALLEL_THRESHOLD=<n>`;
/// runtimes with cheaper team startup (typically libgomp on Linux) should set a
/// considerably lower value.
#ifndef NUMERICS_PARALLEL_THRESHOLD
#define NUMERICS_PARALLEL_THRESHOLD (1 << 18)
#endif
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
