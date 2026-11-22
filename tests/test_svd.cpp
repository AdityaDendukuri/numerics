#include "container/matrix.hpp"
#include "container/vector.hpp"
#include "linear/svd/svd.hpp"
#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>

using namespace num;

// Helpers

static mat<real> make_rect(idx m, idx n) {
    mat<real> A(m, n, 0.0);
    for (idx i = 0; i < m; ++i) {
        for (idx j = 0; j < n; ++j) {
            A(i, j) = 1.0 / (1.0 + i + j);
        }
    }
    return A;
}

/// ||A - U*diag(S)*Vt||_max
static real reconstruction_error(const mat<real> &A, const svd_result &r) {
    idx m = A.rows(), n = A.cols(), k = r.S.size();
    real err = 0.0;
    for (idx i = 0; i < m; ++i) {
        for (idx j = 0; j < n; ++j) {
            real aij = 0.0;
            for (idx p = 0; p < k; ++p) {
                aij += r.U(i, p) * r.S[p] * r.Vt(p, j);
            }
            err = std::max(err, std::abs(A(i, j) - aij));
        }
    }
    return err;
}

/// ||U^T*U - I||_max  (column orthonormality of U)
static real col_ortho_error(const mat<real> &U) {
    idx m = U.rows(), k = U.cols();
    real err = 0.0;
    for (idx i = 0; i < k; ++i) {
        for (idx j = 0; j < k; ++j) {
            real dot = 0.0;
            for (idx p = 0; p < m; ++p) {
                dot += U(p, i) * U(p, j);
            }
            err = std::max(err, std::abs(dot - (i == j ? 1.0 : 0.0)));
        }
    }
    return err;
}

/// Singular values must be non-negative and descending.
static bool sv_descending(const vec<real> &S) {
    for (idx i = 1; i < S.size(); ++i) {
        if (S[i] > S[i - 1] + 1e-12) {
            return false;
        }
    }
    return true;
}

// One-sided Jacobi

TEST(SVD_Jacobi, Reconstruct3x3) {
    mat<real> A = make_rect(3, 3);
    auto r = seq::svd(A, 1e-12, 100);
    EXPECT_TRUE(r.converged);
    EXPECT_LT(reconstruction_error(A, r), 1e-10);
    EXPECT_LT(col_ortho_error(r.U), 1e-10);
    EXPECT_TRUE(sv_descending(r.S));
}

TEST(SVD_Jacobi, ReconstructRectangular) {
    mat<real> A = make_rect(8, 4);
    auto r = seq::svd(A, 1e-12, 100);
    EXPECT_LT(reconstruction_error(A, r), 1e-10);
    EXPECT_LT(col_ortho_error(r.U), 1e-10);
}

TEST(SVD_Jacobi, ReconstructN32) {
    mat<real> A = make_rect(32, 32);
    auto r = seq::svd(A, 1e-12, 100);
    EXPECT_LT(reconstruction_error(A, r), 1e-8);
    EXPECT_LT(col_ortho_error(r.U), 1e-8);
    EXPECT_TRUE(sv_descending(r.S));
}

// LAPACK

#if defined(NUMERICS_HAS_LAPACK)

TEST(SVD_LAPACK, Reconstruct3x3) {
    mat<real> A = make_rect(3, 3);
    auto r = lapack::svd(A);
    EXPECT_TRUE(r.converged);
    EXPECT_LT(reconstruction_error(A, r), 1e-10);
    EXPECT_LT(col_ortho_error(r.U), 1e-10);
    EXPECT_TRUE(sv_descending(r.S));
}

TEST(SVD_LAPACK, MatchesJacobi) {
    mat<real> A = make_rect(16, 16);
    auto rj = seq::svd(A, 1e-12, 100);
    auto rl = lapack::svd(A);
    ASSERT_EQ(rj.S.size(), rl.S.size());
    for (idx i = 0; i < rj.S.size(); ++i) {
        EXPECT_NEAR(rj.S[i], rl.S[i], 1e-8);
    }
}

TEST(SVD_LAPACK, ReconstructN64) {
    mat<real> A = make_rect(64, 64);
    auto r = lapack::svd(A);
    EXPECT_LT(reconstruction_error(A, r), 1e-8);
    EXPECT_LT(col_ortho_error(r.U), 1e-8);
    EXPECT_TRUE(sv_descending(r.S));
}

TEST(SVD_LAPACK, ReconstructRectangular) {
    mat<real> A = make_rect(32, 16);
    auto r = lapack::svd(A);
    EXPECT_LT(reconstruction_error(A, r), 1e-8);
}

#endif // NUMERICS_HAS_LAPACK

// Randomized truncated SVD

static mat<real> make_lowrank(idx n, idx k) {
    mat<real> A(n, n, 0.0);
    for (idx i = 0; i < k; ++i) {
        A(i, i) = std::pow(10.0, static_cast<real>(k - i));
    }
    for (idx i = 0; i < n; ++i) {
        for (idx j = 0; j < n; ++j) {
            A(i, j) += 1e-6 / (1.0 + i + j);
        }
    }
    return A;
}

TEST(SVD_Randomized, TopKSingularValues) {
    // Low-rank matrix with clear spectral gap after k=5: the algorithm is
    // designed to recover singular values that lie above the tail noise floor.
    constexpr idx k = 5;
    mat<real> A = make_lowrank(32, k);
    auto rfull = seq::svd(A, 1e-12, 100);
    auto rrand = svd_truncated(A, k);
    // top-k singular values should match to within 0.1% of the dominant S[0]
    for (idx i = 0; i < k; ++i) {
        EXPECT_NEAR(rrand.S[i], rfull.S[i], rfull.S[0] * 1e-3);
    }
}

TEST(SVD_Randomized, ColumnOrthonormality) {
    mat<real> A = make_rect(20, 20);
    auto r = svd_truncated(A, 8);
    EXPECT_LT(col_ortho_error(r.U), 1e-8);
}

// Singular values from eig(A^T A) lose everything below sqrt(eps) * sigma_max; a proper SVD
// keeps the small ones to absolute accuracy eps * sigma_max.
TEST(SVD, DoesNotSquareTheConditionNumber) {
    constexpr idx n = 4;
    const real sigma[n] = {1.0, 1e-4, 1e-8, 1e-11};
    // Orthogonal factors from two Householder reflectors.
    auto reflector = [](const std::vector<real> &v) {
        real vv = 0;
        for (real x : v) {
            vv += x * x;
        }
        mat<real> P(n, n, 0.0);
        for (idx i = 0; i < n; ++i) {
            for (idx j = 0; j < n; ++j) {
                P(i, j) = (i == j ? 1.0 : 0.0) - 2.0 * v[i] * v[j] / vv;
            }
        }
        return P;
    };
    const mat<real> U = reflector({1, 2, -1, 3});
    const mat<real> V = reflector({2, -1, 1, 1});
    mat<real> A(n, n, 0.0);
    for (idx i = 0; i < n; ++i) {
        for (idx j = 0; j < n; ++j) {
            for (idx k = 0; k < n; ++k) {
                A(i, j) += U(i, k) * sigma[k] * V(j, k);
            }
        }
    }
    for (const auto &[r, which] : {std::pair{svd(A), "default"}, std::pair{seq::svd(A, 1e-15, 100), "seq"}}) {
        ASSERT_EQ(r.S.size(), n) << which;
        for (idx k = 0; k < n; ++k) {
            EXPECT_NEAR(r.S[k], sigma[k], 20.0 * n * eps) << which << " k = " << k;
        }
    }
}
