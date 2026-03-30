/// @file core/backends/seq/matrix.cpp
/// @brief Sequential and blocked C++ matrix operations
///
/// seq:     textbook reference implementations (correct, readable, always available).
/// blocked: cache-blocked loops that compiler can auto-vectorize.
///   matmul_blocked and matmul_register_blocked live here -- they are pure
///   optimization algorithms, not class implementation.

#include "core/matrix.hpp"
#include <algorithm>

namespace num::backends::seq {

void matmul(const Matrix& A, const Matrix& B, Matrix& C) {
    const idx M = A.rows(), K = A.cols(), N = B.cols();
    for (idx i = 0; i < M; ++i)
        for (idx j = 0; j < N; ++j) {
            C(i, j) = 0;
            for (idx k = 0; k < K; ++k)
                C(i, j) += A(i, k) * B(k, j);
        }
}

void matvec(const Matrix& A, const Vector& x, Vector& y) {
    for (idx i = 0; i < A.rows(); ++i) {
        y[i] = 0;
        for (idx j = 0; j < A.cols(); ++j)
            y[i] += A(i, j) * x[j];
    }
}

void matadd(real alpha, const Matrix& A, real beta, const Matrix& B, Matrix& C) {
    for (idx i = 0; i < A.size(); ++i)
        C.data()[i] = alpha * A.data()[i] + beta * B.data()[i];
}

// Cache-blocked matrix multiply
//
// WHY THE NAIVE LOOP IS SLOW
// All three matrices are stored row-major.  In the naive i-j-k order:
//
//   for i:                         <- row of A and C
//     for j:                       <- column of B, element of C
//       for k:                     <- inner dimension
//         C(i,j) += A(i,k)*B(k,j)
//
// A(i,k) is read left-to-right along row i as k increases -- cache-friendly.
// B(k,j) for fixed j and varying k is a stride-N access (column of B) --
// every step jumps N doubles = N*8 bytes.  For N=512 that is 4 KB per step,
// thrashing the cache and stalling on every B load.
//
// HOW BLOCKING FIXES IT
// Divide A, B, C into BLOCK*BLOCK tiles.  The working set for one tile
// triple is  3 * BLOCK^2 * 8 bytes:
//
//   BLOCK = 32  ->  24 KB   (fits in a 32 KB L1 data cache)
//   BLOCK = 64  ->  98 KB   (fits in a 256 KB L2 cache)
//   BLOCK = 128 -> 393 KB   (fits in a 512 KB L2 or L3 at 4 MB)
//
// Within each tile the hot j loop accesses B(k,j) left-to-right along a
// single row of B -- now fully sequential.  The hardware prefetcher can
// recognise the pattern and hide latency.
//
// OUTER LOOP ORDER:  ii -> jj -> kk
// For fixed (ii, jj), the C tile stays in L2 for the entire kk loop.
//
// INNER (MICRO-KERNEL) LOOP ORDER:  i -> k -> j
// A(i,k) is hoisted into a register, turning the inner kernel into:
//   for j in [jj, jj+B):
//     C[i][j] += scalar * B[k][j]   <- AXPY on contiguous memory
// Trivially auto-vectorisable.

void matmul_blocked(const Matrix& A, const Matrix& B, Matrix& C, idx block_size) {
    const idx M = A.rows(), K = A.cols(), N = B.cols();
    std::fill_n(C.data(), M * N, real(0));

    for (idx ii = 0; ii < M; ii += block_size) {
        const idx i_end = std::min(ii + block_size, M);
        for (idx jj = 0; jj < N; jj += block_size) {
            const idx j_end = std::min(jj + block_size, N);
            for (idx kk = 0; kk < K; kk += block_size) {
                const idx k_end = std::min(kk + block_size, K);
                for (idx i = ii; i < i_end; ++i) {
                    for (idx k = kk; k < k_end; ++k) {
                        const real a_ik = A(i, k);
                        for (idx j = jj; j < j_end; ++j)
                            C(i, j) += a_ik * B(k, j);
                    }
                }
            }
        }
    }
}

// Register-blocked matrix multiply
//
// Extends cache blocking with a REG*REG register tile inside each cache tile.
// The accumulator for each small C sub-block is kept in local doubles
// (promoted to registers by the compiler) for the full k sweep, then written
// back to C once -- eliminating the per-iteration load+store of C.
//
// Register blocking only pays off when combined with explicit SIMD, where each
// j-step processes 4 doubles simultaneously.  This file is the conceptual
// bridge: the loop structure here is exactly what matmul_avx implements, with
// scalar loads replaced by vector intrinsics.

void matmul_register_blocked(const Matrix& A, const Matrix& B, Matrix& C,
                              idx block_size, idx reg_size) {
    const idx M = A.rows(), K = A.cols(), N = B.cols();
    std::fill_n(C.data(), M * N, real(0));

    for (idx ii = 0; ii < M; ii += block_size) {
        const idx i_lim = std::min(ii + block_size, M);
        for (idx jj = 0; jj < N; jj += block_size) {
            const idx j_lim = std::min(jj + block_size, N);
            for (idx kk = 0; kk < K; kk += block_size) {
                const idx k_lim = std::min(kk + block_size, K);
                for (idx ir = ii; ir < i_lim; ir += reg_size) {
                    const idx ri = std::min(ir + reg_size, i_lim);
                    for (idx jr = jj; jr < j_lim; jr += reg_size) {
                        const idx rj = std::min(jr + reg_size, j_lim);
                        real c[4][4] = {};
                        for (idx i = ir; i < ri; ++i)
                            for (idx j = jr; j < rj; ++j)
                                c[i - ir][j - jr] = C(i, j);
                        for (idx k = kk; k < k_lim; ++k) {
                            for (idx i = ir; i < ri; ++i) {
                                const real a_ik = A(i, k);
                                for (idx j = jr; j < rj; ++j)
                                    c[i - ir][j - jr] += a_ik * B(k, j);
                            }
                        }
                        for (idx i = ir; i < ri; ++i)
                            for (idx j = jr; j < rj; ++j)
                                C(i, j) = c[i - ir][j - jr];
                    }
                }
            }
        }
    }
}

} // namespace num::backends::seq
