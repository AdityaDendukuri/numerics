/// @file omp/matrix_ops.hpp
/// @brief OpenMP-accelerated Level-2/3 dense matrix operations.
///
/// Same rule as `omp/vector_ops.hpp`: OpenMP only decides how the matrix is
/// sliced into blocks across threads; the arithmetic inside each block is a
/// `num::kernel` call, not a hand-written loop.
///
/// Row-tiled rather than routed through `dispatch::parallel_apply`: that
/// helper's block size and threshold are tuned for vector *element* counts
/// (blocks of ~16K elements, threading only above ~262K elements), which are
/// the wrong units here — a matmul row does O(n*k) work, not O(1), so a matrix
/// with a few hundred rows would never cross an element-counted threshold and
/// would silently never thread. Always parallelizing over row-tiles (as the
/// previous hand-written version did) is the correct granularity for Level-2/3.
#pragma once

#include "container/matrix.hpp"
#include "container/vector.hpp"
#include "core/types.hpp"
#include "kernel/kernel.hpp"
#include <algorithm>

namespace num::omp {

inline void matmul(const mat &A, const mat &B, mat &C) {
    constexpr idx block_size = 64;
    const idx m = A.rows(), k = A.cols(), n = B.cols();
    const real *ad = A.data();
    const real *bd = B.data();
    real *cd = C.data();
#if defined(NUMERICS_HAS_OMP)
#pragma omp parallel for schedule(dynamic)
#endif
    for (idx ii = 0; ii < m; ii += block_size) {
        const idx rows = std::min(block_size, m - ii);
        // The row tile is the only decision made here: `kernel::gemm` does its
        // own register and cache blocking inside each tile.
        kernel::gemm(cd + (ii * n), ad + (ii * k), bd, real(1), real(0), rows, n, k);
    }
}

inline void matvec(const mat &A, const vec &x, vec &y) {
    const idx n = A.cols();
    const real *ad = A.data();
    const real *xd = x.data();
    real *yd = y.data();
#if defined(NUMERICS_HAS_OMP)
#pragma omp parallel for schedule(static)
#endif
    for (idx i = 0; i < A.rows(); ++i) {
        yd[i] = kernel::dot(ad + (i * n), xd, n);
    }
}

inline void matadd(real alpha, const mat &A, real beta, const mat &B, mat &C) {
    constexpr idx block_size = idx{1} << 16;
    const idx total = A.size();
    const real *ad = A.data();
    const real *bd = B.data();
    real *cd = C.data();
#if defined(NUMERICS_HAS_OMP)
#pragma omp parallel for schedule(static)
#endif
    for (idx offset = 0; offset < total; offset += block_size) {
        const idx length = std::min(block_size, total - offset);
        kernel::axpbyz(cd + offset, ad + offset, bd + offset, alpha, beta, length);
    }
}

} // namespace num::omp
