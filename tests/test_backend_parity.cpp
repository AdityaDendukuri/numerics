/// Side-by-side checks: every operation with a BLAS or LAPACK path gives the same answer
/// through that path as through the in-tree implementation. Skipped when the build has no
/// BLAS (or LAPACK), where both calls would run the same in-tree code.

#include "linear/banded/banded.hpp"
#include <gtest/gtest.h>
#include <numerics.hpp>
#include <random>

using namespace num;

namespace {

constexpr real tol = 1e-10;

mat<real> random_matrix(idx m, idx n, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<real> u(-1.0, 1.0);
    mat<real> A(m, n);
    for (idx i = 0; i < m; ++i)
        for (idx j = 0; j < n; ++j)
            A(i, j) = u(rng);
    return A;
}

vec<real> random_vector(idx n, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<real> u(-1.0, 1.0);
    vec<real> v(n);
    for (idx i = 0; i < n; ++i)
        v[i] = u(rng);
    return v;
}

// Diagonally dominant, so LU is well conditioned with or without pivoting.
mat<real> dominant_matrix(idx n, unsigned seed) {
    mat<real> A = random_matrix(n, n, seed);
    for (idx i = 0; i < n; ++i)
        A(i, i) += real(n);
    return A;
}

mat<real> spd_matrix(idx n, unsigned seed) {
    const mat<real> M = random_matrix(n, n, seed);
    mat<real> A(n, n, 0.0);
    for (idx i = 0; i < n; ++i)
        for (idx j = 0; j < n; ++j) {
            for (idx k = 0; k < n; ++k)
                A(i, j) += M(i, k) * M(j, k);
        }
    for (idx i = 0; i < n; ++i)
        A(i, i) += real(n);
    return A;
}

mat<real> lower_triangle(idx n, unsigned seed) {
    mat<real> L = random_matrix(n, n, seed);
    for (idx i = 0; i < n; ++i) {
        for (idx j = i + 1; j < n; ++j)
            L(i, j) = 0.0;
        L(i, i) += 4.0;
    }
    return L;
}

mat<real> upper_triangle(idx n, unsigned seed) {
    mat<real> U = random_matrix(n, n, seed);
    for (idx i = 0; i < n; ++i) {
        for (idx j = 0; j < i; ++j)
            U(i, j) = 0.0;
        U(i, i) += 4.0;
    }
    return U;
}

band_mat random_band(idx n, idx kl, idx ku, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<real> u(-1.0, 1.0);
    band_mat A(n, kl, ku);
    for (idx j = 0; j < n; ++j)
        for (idx i = (j > ku ? j - ku : 0); i <= std::min(n - 1, j + kl); ++i)
            A(i, j) = u(rng) + (i == j ? real(kl + ku + 2) : 0.0);
    return A;
}

void expect_near(const vec<real> &a, const vec<real> &b, real t = tol) {
    ASSERT_EQ(a.size(), b.size());
    for (idx i = 0; i < a.size(); ++i)
        EXPECT_NEAR(a[i], b[i], t) << "at " << i;
}

void expect_near(const mat<real> &A, const mat<real> &B, real t = tol) {
    ASSERT_EQ(A.rows(), B.rows());
    ASSERT_EQ(A.cols(), B.cols());
    for (idx i = 0; i < A.rows(); ++i)
        for (idx j = 0; j < A.cols(); ++j)
            EXPECT_NEAR(A(i, j), B(i, j), t) << "at (" << i << ", " << j << ")";
}

#define REQUIRE_BLAS()                                                                             \
    if constexpr (!has_blas) {                                                                     \
        GTEST_SKIP() << "built without BLAS";                                                      \
    }
#define REQUIRE_LAPACK()                                                                           \
    if constexpr (!has_lapack) {                                                                   \
        GTEST_SKIP() << "built without LAPACK";                                                    \
    }

} // namespace

// BLAS level 1

TEST(BlasParity, VectorOperations) {
    REQUIRE_BLAS();
    const vec<real> x = random_vector(257, 1), y = random_vector(257, 2);
    EXPECT_NEAR(blas::dot(x, y), seq::dot(x, y), tol);
    EXPECT_NEAR(blas::norm(x), seq::norm(x), tol);

    vec<real> a = y, b = y;
    blas::axpy(0.7, x, a);
    seq::axpy(0.7, x, b);
    expect_near(a, b);

    blas::scale(a, -1.3);
    seq::scale(b, -1.3);
    expect_near(a, b);

    vec<real> za(257), zb(257);
    blas::add(x, y, za);
    seq::add(x, y, zb);
    expect_near(za, zb);
}

// BLAS level 2

TEST(BlasParity, MatrixVectorProduct) {
    REQUIRE_BLAS();
    const mat<real> A = random_matrix(37, 23, 3);
    const vec<real> x = random_vector(23, 4);
    vec<real> ya(37), yb(37);
    blas::matvec(A, x, ya);
    seq::matvec(A, x, yb);
    expect_near(ya, yb);
}

TEST(BlasParity, RankOneUpdate) {
    REQUIRE_BLAS();
    const vec<real> x = random_vector(31, 5), y = random_vector(19, 6);
    mat<real> A = random_matrix(31, 19, 7), B = A;
    blas::ger(0.4, x, y, A);
    seq::ger(0.4, x, y, B);
    expect_near(A, B);
}

TEST(BlasParity, TriangularSolveVector) {
    REQUIRE_BLAS();
    const idx n = 41;
    const vec<real> b = random_vector(n, 8);
    vec<real> xa, xb;
    blas::trsv_lower(lower_triangle(n, 9), b, xa);
    seq::trsv_lower(lower_triangle(n, 9), b, xb);
    expect_near(xa, xb);
    blas::trsv_upper(upper_triangle(n, 10), b, xa);
    seq::trsv_upper(upper_triangle(n, 10), b, xb);
    expect_near(xa, xb);
}

TEST(BlasParity, BandedMatrixVectorProduct) {
    REQUIRE_BLAS();
    const band_mat A = random_band(53, 3, 5, 11);
    const vec<real> x = random_vector(53, 12);
    vec<real> ya = random_vector(53, 13), yb = ya;
    blas::banded_gemv(1.5, A, x, -0.5, ya);
    seq::banded_gemv(1.5, A, x, -0.5, yb);
    expect_near(ya, yb);
}

// BLAS level 3

TEST(BlasParity, MatrixProduct) {
    REQUIRE_BLAS();
    const mat<real> A = random_matrix(29, 17, 14), B = random_matrix(17, 23, 15);
    mat<real> Ca(29, 23), Cb(29, 23);
    blas::matmul(A, B, Ca);
    seq::matmul(A, B, Cb);
    expect_near(Ca, Cb);
}

TEST(BlasParity, GeneralMatrixProductAllTransposes) {
    REQUIRE_BLAS();
    const idx m = 13, n = 11, k = 17;
    for (const bool ta : {false, true})
        for (const bool tb : {false, true}) {
            const mat<real> A = ta ? random_matrix(k, m, 16) : random_matrix(m, k, 16);
            const mat<real> B = tb ? random_matrix(n, k, 17) : random_matrix(k, n, 17);
            mat<real> C = random_matrix(m, n, 18), reference = C;
            blas::gemm(0.8, A, ta, B, tb, 0.3, C);
            for (idx i = 0; i < m; ++i)
                for (idx j = 0; j < n; ++j) {
                    real s = 0.0;
                    for (idx p = 0; p < k; ++p)
                        s += (ta ? A(p, i) : A(i, p)) * (tb ? B(j, p) : B(p, j));
                    reference(i, j) = (0.8 * s) + (0.3 * reference(i, j));
                }
            expect_near(C, reference);
        }
}

TEST(BlasParity, SymmetricRankKUpdate) {
    REQUIRE_BLAS();
    const idx rows = 21, cols = 9;
    const mat<real> A = random_matrix(rows, cols, 19);
    mat<real> Ca = spd_matrix(rows, 20), Cb = Ca;
    blas::syrk_lower(0.6, A.data(), cols, 0.2, Ca.data(), rows, rows, cols);
    kernel::syrk_lower(Cb.data(), rows, A.data(), cols, 0.6, 0.2, rows, cols);
    for (idx i = 0; i < rows; ++i)
        for (idx j = 0; j <= i; ++j)
            EXPECT_NEAR(Ca(i, j), Cb(i, j), tol);
}

TEST(BlasParity, TriangularSolveMatrix) {
    REQUIRE_BLAS();
    const idx n = 33, nrhs = 7;
    const mat<real> L = lower_triangle(n, 21), U = upper_triangle(n, 22);
    const mat<real> B = random_matrix(n, nrhs, 23);
    struct Case {
        const mat<real> &T;
        bool upper, transpose;
    };
    for (const Case c : {Case{L, false, false}, Case{L, false, true}, Case{U, true, false},
                         Case{U, true, true}}) {
        mat<real> Xa = B, Xb = B;
        blas::trsm_left(c.T.data(), n, Xa.data(), nrhs, n, nrhs, c.upper, c.transpose, false);
        if (!c.upper && !c.transpose)
            kernel::trsm_lower_inplace(Xb.data(), nrhs, c.T.data(), n, nrhs);
        else if (!c.upper)
            kernel::trsm_lower_transpose_inplace(Xb.data(), nrhs, c.T.data(), n, nrhs);
        else if (!c.transpose)
            kernel::trsm_upper_inplace(Xb.data(), nrhs, c.T.data(), n, n, nrhs);
        else
            kernel::trsm_upper_transpose_inplace(Xb.data(), nrhs, c.T.data(), n, n, nrhs);
        expect_near(Xa, Xb);
    }

    mat<real> Xa = random_matrix(nrhs, n, 24), Xb = Xa;
    blas::trsm_right_lower_transpose(L.data(), n, Xa.data(), n, nrhs, n);
    kernel::trsm_lower_transpose_right_inplace(Xb.data(), n, L.data(), n, nrhs, n);
    expect_near(Xa, Xb);
}

// LAPACK

TEST(LapackParity, LU) {
    REQUIRE_LAPACK();
    const idx n = 45;
    const mat<real> A = dominant_matrix(n, 25);
    const auto fa = lapack::lu(A), fb = seq::lu(A);
    expect_near(fa.LU, fb.LU);
    EXPECT_EQ(fa.swaps, fb.swaps);

    const vec<real> b = random_vector(n, 26);
    vec<real> xa, xb;
    lapack::solve(fb, b, xa);
    num::solve(fb, b, xb);
    expect_near(xa, xb);

    const mat<real> B = random_matrix(n, 5, 27);
    mat<real> Xa, Xb;
    lapack::solve(fb, B, Xa);
    num::solve(fb, B, Xb);
    expect_near(Xa, Xb);

    expect_near(lapack::inverse(fb), seq::inverse(fb));
}

TEST(LapackParity, Cholesky) {
    REQUIRE_LAPACK();
    const idx n = 39;
    const mat<real> A = spd_matrix(n, 28);
    const auto fa = lapack::cholesky(A), fb = unsafe::cholesky(A);
    ASSERT_TRUE(fa.positive_definite && fb.positive_definite);
    expect_near(fa.L, fb.L);

    const vec<real> b = random_vector(n, 29);
    vec<real> xa, xb;
    lapack::solve(fb, b, xa);
    num::solve(fb, b, xb);
    expect_near(xa, xb);

    const mat<real> B = random_matrix(n, 4, 30);
    mat<real> Xa, Xb;
    lapack::solve(fb, B, Xa);
    num::solve(fb, B, Xb);
    expect_near(Xa, Xb);
}

TEST(LapackParity, QR) {
    REQUIRE_LAPACK();
    // Q and R are unique up to the signs of R's diagonal, so compare least-squares solutions.
    const mat<real> A = random_matrix(40, 25, 31);
    const vec<real> b = random_vector(40, 32);
    vec<real> xa, xb;
    num::solve(lapack::qr(A), b, xa);
    num::solve(seq::qr(A), b, xb);
    xa = vec<real>(view<const real>(xa.data(), 25));
    xb = vec<real>(view<const real>(xb.data(), 25));
    expect_near(xa, xb, 1e-9);
}

TEST(LapackParity, Hessenberg) {
    REQUIRE_LAPACK();
    // H is unique up to signs; both must reproduce A = Q H Q^T with H upper Hessenberg.
    const idx n = 30;
    const mat<real> A = random_matrix(n, n, 33);
    for (const auto &d : {lapack::hessenberg(A), seq::hessenberg(A)}) {
        mat<real> QH(n, n, 0.0), R(n, n, 0.0);
        for (idx i = 0; i < n; ++i)
            for (idx j = 0; j < n; ++j)
                for (idx k = 0; k < n; ++k)
                    QH(i, j) += d.Q()(i, k) * d.H()(k, j);
        for (idx i = 0; i < n; ++i)
            for (idx j = 0; j < n; ++j)
                for (idx k = 0; k < n; ++k)
                    R(i, j) += QH(i, k) * d.Q()(j, k);
        expect_near(R, A, 1e-9);
        for (idx i = 2; i < n; ++i)
            for (idx j = 0; j + 1 < i; ++j)
                EXPECT_NEAR(d.H()(i, j), 0.0, tol);
    }
}

TEST(LapackParity, SymmetricEigenvalues) {
    REQUIRE_LAPACK();
    const mat<real> A = spd_matrix(24, 34);
    expect_near(lapack::eig_sym(A).values, seq::eig_sym(A, 1e-13, 200).values, 1e-8);
}

TEST(LapackParity, SingularValues) {
    REQUIRE_LAPACK();
    const mat<real> A = random_matrix(26, 18, 35);
    expect_near(lapack::svd(A).S, seq::svd(A, 1e-13, 200).S, 1e-9);
}

TEST(LapackParity, Tridiagonal) {
    REQUIRE_LAPACK();
    const idx n = 50;
    vec<real> a = random_vector(n - 1, 36), c = random_vector(n - 1, 37), b(n, 4.0);
    const vec<real> d = random_vector(n, 38);
    vec<real> xa(n), xb(n);
    lapack::thomas(a, b, c, d, xa);
    seq::thomas(a, b, c, d, xb);
    expect_near(xa, xb);
}

TEST(LapackParity, BandedLU) {
    REQUIRE_LAPACK();
    const band_mat A = random_band(61, 4, 2, 39);
    const auto fa = lapack::lu(A), fb = seq::lu(A);
    EXPECT_EQ(fa.swaps, fb.swaps);

    const vec<real> b = random_vector(61, 40);
    vec<real> xa, xb, xc;
    num::solve(fa, b, xa);
    num::solve(fb, b, xb);
    lapack::solve(fb, b, xc);
    expect_near(xa, xb);
    expect_near(xc, xb);

    const mat<real> B = random_matrix(61, 3, 41);
    mat<real> Xa, Xb;
    lapack::solve(fb, B, Xa);
    num::solve(fb, B, Xb);
    expect_near(Xa, Xb);
}
