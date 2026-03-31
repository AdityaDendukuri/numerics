/// @file core/backends/omp/matrix.cpp
/// @brief OpenMP backend  -- parallelised matrix operations
///
/// The (ii, jj) cache-tile pairs are distributed across threads.
/// Each thread owns a distinct C tile so there are no data races.
/// Falls back to sequential when NUMERICS_HAS_OMP is not defined.

#include "core/matrix.hpp"
#include "../seq/impl.hpp"
#include <algorithm>

namespace num::backends::omp {

void matmul(const Matrix& A, const Matrix& B, Matrix& C) {
#ifdef NUMERICS_HAS_OMP
    constexpr idx BS = 64;
    const idx     M = A.rows(), K = A.cols(), N = B.cols();
    std::fill_n(C.data(), M * N, real(0));

    #pragma omp parallel for schedule(dynamic) collapse(2)
    for (idx ii = 0; ii < M; ii += BS) {
        for (idx jj = 0; jj < N; jj += BS) {
            const idx i_lim = std::min(ii + BS, M);
            const idx j_lim = std::min(jj + BS, N);
            for (idx kk = 0; kk < K; kk += BS) {
                const idx k_lim = std::min(kk + BS, K);
                for (idx i = ii; i < i_lim; ++i) {
                    for (idx k = kk; k < k_lim; ++k) {
                        const real a_ik = A(i, k);
                        for (idx j = jj; j < j_lim; ++j)
                            C(i, j) += a_ik * B(k, j);
                    }
                }
            }
        }
    }
#else
    num::backends::seq::matmul(A, B, C);
#endif
}

void matvec(const Matrix& A, const Vector& x, Vector& y) {
#ifdef NUMERICS_HAS_OMP
    #pragma omp parallel for schedule(static)
    for (idx i = 0; i < A.rows(); ++i) {
        real sum = 0;
        for (idx j = 0; j < A.cols(); ++j)
            sum += A(i, j) * x[j];
        y[i] = sum;
    }
#else
    num::backends::seq::matvec(A, x, y);
#endif
}

void matadd(real          alpha,
            const Matrix& A,
            real          beta,
            const Matrix& B,
            Matrix&       C) {
#ifdef NUMERICS_HAS_OMP
    const idx   n = A.size();
    #pragma omp parallel for schedule(static)
    for (idx i = 0; i < n; ++i)
        C.data()[i] = alpha * A.data()[i] + beta * B.data()[i];
#else
    num::backends::seq::matadd(alpha, A, beta, B, C);
#endif
}

} // namespace num::backends::omp
