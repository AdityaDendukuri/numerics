#include <gtest/gtest.h>
#include "linalg/svd/svd.hpp"
#include "core/matrix.hpp"
#include "core/vector.hpp"
#include <cmath>
#include <algorithm>

using namespace num;

// ── Helpers ──────────────────────────────────────────────────────────────────

static Matrix make_rect(idx m, idx n) {
    Matrix A(m, n, 0.0);
    for (idx i = 0; i < m; ++i)
        for (idx j = 0; j < n; ++j)
            A(i, j) = 1.0 / (1.0 + i + j);
    return A;
}

/// ||A - U*diag(S)*Vt||_max
static real reconstruction_error(const Matrix& A, const SVDResult& r) {
    idx m = A.rows(), n = A.cols(), k = r.S.size();
    real err = 0.0;
    for (idx i = 0; i < m; ++i)
        for (idx j = 0; j < n; ++j) {
            real aij = 0.0;
            for (idx p = 0; p < k; ++p)
                aij += r.U(i, p) * r.S[p] * r.Vt(p, j);
            err = std::max(err, std::abs(A(i, j) - aij));
        }
    return err;
}

/// ||U^T*U - I||_max  (column orthonormality of U)
static real col_ortho_error(const Matrix& U) {
    idx m = U.rows(), k = U.cols();
    real err = 0.0;
    for (idx i = 0; i < k; ++i)
        for (idx j = 0; j < k; ++j) {
            real dot = 0.0;
            for (idx p = 0; p < m; ++p) dot += U(p, i) * U(p, j);
            err = std::max(err, std::abs(dot - (i == j ? 1.0 : 0.0)));
        }
    return err;
}

/// Singular values must be non-negative and descending.
static bool sv_descending(const Vector& S) {
    for (idx i = 1; i < S.size(); ++i)
        if (S[i] > S[i-1] + 1e-12) return false;
    return true;
}

// ── One-sided Jacobi (Backend::seq) ─────────────────────────────────────────

TEST(SVD_Jacobi, Reconstruct3x3) {
    Matrix A = make_rect(3, 3);
    auto r = svd(A, Backend::seq);
    EXPECT_TRUE(r.converged);
    EXPECT_LT(reconstruction_error(A, r), 1e-10);
    EXPECT_LT(col_ortho_error(r.U), 1e-10);
    EXPECT_TRUE(sv_descending(r.S));
}

TEST(SVD_Jacobi, ReconstructRectangular) {
    Matrix A = make_rect(8, 4);
    auto r = svd(A, Backend::seq);
    EXPECT_LT(reconstruction_error(A, r), 1e-10);
    EXPECT_LT(col_ortho_error(r.U), 1e-10);
}

TEST(SVD_Jacobi, ReconstructN32) {
    Matrix A = make_rect(32, 32);
    auto r = svd(A, Backend::seq);
    EXPECT_LT(reconstruction_error(A, r), 1e-8);
    EXPECT_LT(col_ortho_error(r.U), 1e-8);
    EXPECT_TRUE(sv_descending(r.S));
}

// ── LAPACK (Backend::lapack) ─────────────────────────────────────────────────

#if defined(NUMERICS_HAS_LAPACK)

TEST(SVD_LAPACK, Reconstruct3x3) {
    Matrix A = make_rect(3, 3);
    auto r = svd(A, Backend::lapack);
    EXPECT_TRUE(r.converged);
    EXPECT_LT(reconstruction_error(A, r), 1e-10);
    EXPECT_LT(col_ortho_error(r.U), 1e-10);
    EXPECT_TRUE(sv_descending(r.S));
}

TEST(SVD_LAPACK, MatchesJacobi) {
    Matrix A = make_rect(16, 16);
    auto rj = svd(A, Backend::seq);
    auto rl = svd(A, Backend::lapack);
    ASSERT_EQ(rj.S.size(), rl.S.size());
    for (idx i = 0; i < rj.S.size(); ++i)
        EXPECT_NEAR(rj.S[i], rl.S[i], 1e-8);
}

TEST(SVD_LAPACK, ReconstructN64) {
    Matrix A = make_rect(64, 64);
    auto r = svd(A, Backend::lapack);
    EXPECT_LT(reconstruction_error(A, r), 1e-8);
    EXPECT_LT(col_ortho_error(r.U), 1e-8);
    EXPECT_TRUE(sv_descending(r.S));
}

TEST(SVD_LAPACK, ReconstructRectangular) {
    Matrix A = make_rect(32, 16);
    auto r = svd(A, Backend::lapack);
    EXPECT_LT(reconstruction_error(A, r), 1e-8);
}

#endif // NUMERICS_HAS_LAPACK

// ── Randomized truncated SVD ──────────────────────────────────────────────────

// Build a rank-k + noise matrix so the randomized SVD has a clear spectral gap
// to exploit.  Dominant singular values are s_i = 10^(k-i) for i=0..k-1; the
// remaining directions are filled with small noise (~1e-6).
static Matrix make_lowrank(idx n, idx k) {
    // U: random-ish orthogonal factor (use Hilbert columns, then orthonormalize via QR)
    // For simplicity use the first k standard basis vectors scaled by s_i.
    // A = sum_i s_i * e_i * e_i^T  (diagonal, trivially rank-k)
    Matrix A(n, n, 0.0);
    for (idx i = 0; i < k; ++i)
        A(i, i) = std::pow(10.0, static_cast<real>(k - i));  // 1e5, 1e4, ..., 10
    // Add small noise so the matrix is dense and non-trivial
    for (idx i = 0; i < n; ++i)
        for (idx j = 0; j < n; ++j)
            A(i, j) += 1e-6 / (1.0 + i + j);
    return A;
}

TEST(SVD_Randomized, TopKSingularValues) {
    // Low-rank matrix with clear spectral gap after k=5: the algorithm is designed
    // to recover singular values that lie above the tail noise floor.
    constexpr idx k = 5;
    Matrix A = make_lowrank(32, k);
    auto rfull = svd(A, Backend::seq);
    auto rrand = svd_truncated(A, k, default_backend);
    // top-k singular values should match to within 0.1% of the dominant S[0]
    for (idx i = 0; i < k; ++i)
        EXPECT_NEAR(rrand.S[i], rfull.S[i], rfull.S[0] * 1e-3);
}

TEST(SVD_Randomized, ColumnOrthonormality) {
    Matrix A = make_rect(20, 20);
    auto r = svd_truncated(A, 8, default_backend);
    EXPECT_LT(col_ortho_error(r.U), 1e-8);
}
