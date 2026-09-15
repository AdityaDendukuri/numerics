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
    // The kernel's own blocked loop (see `kernel::gemm_config`), with the
    // packed A slab and B panel shared by every thread rather than repacked
    // per row tile. Threads split the packing by tile and the microkernel
    // sweep by column tile, which is the BLIS "jr-loop" parallelization; the
    // work-sharing barriers keep the shared panels consistent.
    using cfg = kernel::gemm_config<real>;
    const idx m = A.rows(), k = A.cols(), n = B.cols();
    const real *ad = A.data();
    const real *bd = B.data();
    real *cd = C.data();
    real *work = kernel::detail::gemm_static_workspace<real>();
    real *Ap = work;
    real *Bp =
        work + (kernel::detail::round_up(std::min(m, cfg::mc), cfg::mr) * std::min(k, cfg::kc));

#if defined(NUMERICS_HAS_OMP)
#pragma omp parallel
#endif
    {
#if defined(NUMERICS_HAS_OMP)
#pragma omp for schedule(static)
#endif
        for (idx i = 0; i < m; ++i) {
            kernel::fill(cd + (i * n), real(0), n);
        }
        for (idx jc = 0; jc < n; jc += cfg::nc) {
            const idx nb = std::min(cfg::nc, n - jc);
            for (idx pc = 0; pc < k; pc += cfg::kc) {
                const idx kb = std::min(cfg::kc, k - pc);
#if defined(NUMERICS_HAS_OMP)
#pragma omp for schedule(static)
#endif
                for (idx j0 = 0; j0 < nb; j0 += cfg::nr) {
                    kernel::detail::gemm_pack_b(Bp + (j0 * kb), bd + (pc * n) + jc + j0, n, idx{1},
                                                kb, std::min(cfg::nr, nb - j0));
                }
                for (idx ic = 0; ic < m; ic += cfg::mc) {
                    const idx mb = std::min(cfg::mc, m - ic);
#if defined(NUMERICS_HAS_OMP)
#pragma omp for schedule(static)
#endif
                    for (idx i0 = 0; i0 < mb; i0 += cfg::mr) {
                        kernel::detail::gemm_pack_a(Ap + (i0 * kb), ad + ((ic + i0) * k) + pc, k,
                                                    idx{1}, real(1), std::min(cfg::mr, mb - i0),
                                                    kb);
                    }
#if defined(NUMERICS_HAS_OMP)
#pragma omp for schedule(static)
#endif
                    for (idx jr = 0; jr < nb; jr += cfg::nr) {
                        const idx cols = std::min(cfg::nr, nb - jr);
                        for (idx ir = 0; ir < mb; ir += cfg::mr) {
                            kernel::detail::gemm_micro(cd + ((ic + ir) * n) + jc + jr, n,
                                                       Ap + (ir * kb), Bp + (jr * kb), kb,
                                                       std::min(cfg::mr, mb - ir), cols);
                        }
                    }
                }
            }
        }
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
