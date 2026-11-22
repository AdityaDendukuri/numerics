#include "linear/matrix_utils.hpp"
#include "container/matrix.hpp"
#include "container/matrix_ops.hpp"
#include "container/vector.hpp"
#include "linear/eigen/eigen.hpp"
#include "operator/operator.hpp"

#include <cmath>
#include <limits>
#include <vector>
#include <gtest/gtest.h>

using namespace num;

template <class Op>
concept LanczosCallable = requires(const Op &A) {
    lanczos(A, 2);
};

static_assert(!LanczosCallable<operators::dense_op>);

// Helpers

static mat<real> make_sym(idx n) {
    mat<real> A(n, n, 0.0);
    for (idx i = 0; i < n; ++i) {
        for (idx j = i; j < n; ++j) {
            real v = 1.0 / (1.0 + i + j);
            A(i, j) = A(j, i) = v;
        }
    }
    for (idx i = 0; i < n; ++i) {
        A(i, i) += static_cast<real>(n);
    }
    return A;
}

static real reconstruction_error(const mat<real> &A, const eigen_result &r) {
    idx n = A.rows();
    real err = 0.0;
    for (idx i = 0; i < n; ++i) {
        for (idx j = 0; j < n; ++j) {
            real aij = 0.0;
            for (idx k = 0; k < n; ++k) {
                aij += r.vectors(i, k) * r.values[k] * r.vectors(j, k);
            }
            err = std::max(err, std::abs(A(i, j) - aij));
        }
    }
    return err;
}

static real orthogonality_error(const mat<real> &V) {
    idx n = V.rows();
    real err = 0.0;
    for (idx i = 0; i < n; ++i) {
        for (idx j = 0; j < n; ++j) {
            real dot = 0.0;
            for (idx k = 0; k < n; ++k) {
                dot += V(k, i) * V(k, j);
            }
            real expected = (i == j) ? 1.0 : 0.0;
            err = std::max(err, std::abs(dot - expected));
        }
    }
    return err;
}

// Jacobi

TEST(EigSym_Jacobi, Reconstruct3x3) {
    mat<real> A(3, 3, 0.0);
    A(0, 0) = 4;
    A(0, 1) = 1;
    A(0, 2) = 0;
    A(1, 0) = 1;
    A(1, 1) = 3;
    A(1, 2) = 1;
    A(2, 0) = 0;
    A(2, 1) = 1;
    A(2, 2) = 2;
    auto r = seq::eig_sym(A, 1e-12, 100);
    EXPECT_TRUE(r.converged);
    EXPECT_LT(reconstruction_error(A, r), 1e-10);
    EXPECT_LT(orthogonality_error(r.vectors), 1e-10);
}

TEST(EigSym_Jacobi, EigenvaluesAscending) {
    mat<real> A = make_sym(8);
    auto r = seq::eig_sym(A, 1e-12, 100);
    for (idx i = 1; i < r.values.size(); ++i) {
        EXPECT_LE(r.values[i - 1], r.values[i] + 1e-12);
    }
}

TEST(EigSym_Jacobi, ReconstructN32) {
    mat<real> A = make_sym(32);
    auto r = seq::eig_sym(A, 1e-12, 100);
    EXPECT_LT(reconstruction_error(A, r), 1e-8);
    EXPECT_LT(orthogonality_error(r.vectors), 1e-8);
}

// LAPACK

#if defined(NUMERICS_HAS_LAPACK)

TEST(EigSym_LAPACK, Reconstruct3x3) {
    mat<real> A(3, 3, 0.0);
    A(0, 0) = 4;
    A(0, 1) = 1;
    A(0, 2) = 0;
    A(1, 0) = 1;
    A(1, 1) = 3;
    A(1, 2) = 1;
    A(2, 0) = 0;
    A(2, 1) = 1;
    A(2, 2) = 2;
    auto r = lapack::eig_sym(A);
    EXPECT_TRUE(r.converged);
    EXPECT_LT(reconstruction_error(A, r), 1e-10);
    EXPECT_LT(orthogonality_error(r.vectors), 1e-10);
}

TEST(EigSym_LAPACK, MatchesJacobi) {
    mat<real> A = make_sym(20);
    auto rj = seq::eig_sym(A, 1e-12, 100);
    auto rl = lapack::eig_sym(A);
    ASSERT_EQ(rj.values.size(), rl.values.size());
    for (idx i = 0; i < rj.values.size(); ++i) {
        EXPECT_NEAR(rj.values[i], rl.values[i], 1e-8);
    }
}

TEST(EigSym_LAPACK, ReconstructN64) {
    mat<real> A = make_sym(64);
    auto r = lapack::eig_sym(A);
    EXPECT_LT(reconstruction_error(A, r), 1e-8);
    EXPECT_LT(orthogonality_error(r.vectors), 1e-8);
}

#endif // NUMERICS_HAS_LAPACK

// Power iteration

TEST(PowerIteration, DominantEigenvalue) {
    // Diagonal matrix: dominant eigenvalue = 10
    idx n = 5;
    mat<real> A(n, n, 0.0);
    A(0, 0) = 10;
    A(1, 1) = 5;
    A(2, 2) = 3;
    A(3, 3) = 2;
    A(4, 4) = 1;
    auto r = power_iteration(A, 1e-10, 1000);
    EXPECT_TRUE(r.converged);
    EXPECT_NEAR(std::abs(r.eigenvalue), 10.0, 1e-8);
}

// Lanczos

TEST(Lanczos, TopKEigenvalues) {
    idx n = 50;
    mat<real> A = make_sym(n);
    auto op = operators::make_op([&](const vec<real> &v, vec<real> &w) { matvec(A, v, w); }, n);
    auto r = lanczos(num::assume_symmetric(op), 5, 1e-10);
    EXPECT_TRUE(r.converged);

    // Compare against Jacobi for top 5 eigenvalues
    auto ref = seq::eig_sym(A, 1e-12, 100);
    for (idx i = 0; i < 5; ++i) {
        real lref = ref.values[n - 1 - i]; // largest first from Lanczos
        bool found = false;
        for (idx j = 0; j < r.ritz_values.size(); ++j) {
            if (std::abs(r.ritz_values[j] - lref) < 1e-4) {
                found = true;
                break;
            }
        }
        EXPECT_TRUE(found) << "Lanczos missed eigenvalue " << lref;
    }
}

TEST(Lanczos, DenseOperator) {
    idx n = 50;
    mat<real> A = make_sym(n);
    operators::dense_op op(A);
    static_assert(self_adjoint_operator<decltype(num::assume_symmetric(op))>);
    auto r = lanczos(num::assume_symmetric(op), 5, 1e-10);
    EXPECT_TRUE(r.converged);

    auto ref = seq::eig_sym(A, 1e-12, 100);
    for (idx i = 0; i < 5; ++i) {
        real lref = ref.values[n - 1 - i];
        bool found = false;
        for (idx j = 0; j < r.ritz_values.size(); ++j) {
            if (std::abs(r.ritz_values[j] - lref) < 1e-4) {
                found = true;
                break;
            }
        }
        EXPECT_TRUE(found) << "Lanczos missed eigenvalue " << lref;
    }
}

TEST(Lanczos, InverseSquareRootAction) {
    constexpr idx n = 12;
    mat<real> A(n, n, 0.0);
    vec<real> right_hand_side(n, 0.0);
    for (idx j = 0; j < n; ++j) {
        A(j, j) = 1.0 + static_cast<real>(j);
        right_hand_side[j] = static_cast<real>(j + 1);
    }

    const auto result = inverse_sqrt_lanczos(num::assume_spd(operators::dense_op(A)),
                                             right_hand_side, 1e-12, n);

    EXPECT_TRUE(result.converged);
    for (idx j = 0; j < n; ++j)
        EXPECT_NEAR(result.value[j], right_hand_side[j] / std::sqrt(A(j, j)), 1e-9);
}

TEST(Lanczos, InverseSquareRootOfZeroVector) {
    mat<real> A(4, 4, 0.0);
    for (idx j = 0; j < 4; ++j)
        A(j, j) = 2.0;

    const auto result =
        inverse_sqrt_lanczos(num::assume_spd(operators::dense_op(A)), vec<real>(4, 0.0));

    EXPECT_TRUE(result.converged);
    EXPECT_EQ(result.steps, 0u);
    EXPECT_EQ(norm(result.value), 0.0);
}

TEST(Lanczos, SquareRootAction) {
    constexpr idx n = 12;
    mat<real> A(n, n, 0.0);
    vec<real> right_hand_side(n, 0.0);
    for (idx j = 0; j < n; ++j) {
        A(j, j) = 1.0 + static_cast<real>(j);
        right_hand_side[j] = static_cast<real>(j + 1);
    }

    const auto result =
        sqrt_lanczos(num::assume_spd(operators::dense_op(A)), right_hand_side, 1e-12, n);

    EXPECT_TRUE(result.converged);
    for (idx j = 0; j < n; ++j)
        EXPECT_NEAR(result.value[j], right_hand_side[j] * std::sqrt(A(j, j)), 1e-9);
}

TEST(Lanczos, SquareRootOfZeroVector) {
    mat<real> A(4, 4, 0.0);
    for (idx j = 0; j < 4; ++j)
        A(j, j) = 2.0;

    const auto result = sqrt_lanczos(num::assume_spd(operators::dense_op(A)), vec<real>(4, 0.0));

    EXPECT_TRUE(result.converged);
    EXPECT_EQ(result.steps, 0u);
    EXPECT_EQ(norm(result.value), 0.0);
}

namespace {

vec<real> dense_times(const mat<real> &A, const vec<real> &x) {
    vec<real> y(A.rows(), 0.0);
    for (idx i = 0; i < A.rows(); ++i) {
        for (idx j = 0; j < A.cols(); ++j) {
            y[i] += A(i, j) * x[j];
        }
    }
    return y;
}

real orthogonality_loss(const mat<real> &Q) {
    real worst = 0;
    for (idx a = 0; a < Q.cols(); ++a) {
        for (idx b = 0; b < Q.cols(); ++b) {
            real s = 0;
            for (idx i = 0; i < Q.rows(); ++i) {
                s += Q(i, a) * Q(i, b);
            }
            worst = std::max(worst, std::abs(s - (a == b ? 1.0 : 0.0)));
        }
    }
    return worst;
}

std::vector<long double> sturm_eigenvalues(const std::vector<long double> &d,
                                           const std::vector<long double> &e) {
    const idx n = d.size();
    auto count_below = [&](long double x) {
        idx count = 0;
        long double q = d[0] - x;
        count += q < 0;
        for (idx i = 1; i < n; ++i) {
            if (q == 0) {
                q = std::numeric_limits<long double>::epsilon();
            }
            q = d[i] - x - e[i - 1] * e[i - 1] / q;
            count += q < 0;
        }
        return count;
    };
    std::vector<long double> values(n);
    for (idx k = 0; k < n; ++k) {
        long double lo = -100, hi = 100;
        for (int it = 0; it < 200; ++it) {
            const long double mid = (lo + hi) / 2;
            (count_below(mid) > k ? hi : lo) = mid;
        }
        values[k] = (lo + hi) / 2;
    }
    return values;
}

mat<real> scaled_laplacian(idx n, real s) {
    mat<real> A(n, n, 0.0);
    for (idx i = 0; i < n; ++i) {
        A(i, i) = s * 2.1;
        if (i + 1 < n) {
            A(i, i + 1) = A(i + 1, i) = -s;
        }
    }
    return A;
}

} // namespace

// Wilkinson's W21+: its top two eigenvalues agree to about 1e-14, so each eigenvector is
// ill-determined, yet a stable symmetric solver returns both values and an orthonormal basis.
TEST(EigSym, WilkinsonW21PlusResolvesNearlyDoubleEigenvalues) {
    constexpr idx n = 21;
    mat<real> W(n, n, 0.0);
    std::vector<long double> d(n), e(n - 1, 1.0L);
    for (idx i = 0; i < n; ++i) {
        d[i] = std::abs(static_cast<long double>(i) - 10.0L);
        W(i, i) = static_cast<real>(d[i]);
        if (i + 1 < n) {
            W(i, i + 1) = W(i + 1, i) = 1.0;
        }
    }
    const auto reference = sturm_eigenvalues(d, e);
    ASSERT_LT(reference[n - 1] - reference[n - 2], 1e-12L);

    auto check = [&](const eigen_result &r, const char *which) {
        ASSERT_EQ(r.values.size(), n) << which;
        for (idx k = 0; k < n; ++k) {
            EXPECT_NEAR(r.values[k], static_cast<real>(reference[k]), 50.0 * eps * 12.0)
                << which << " k = " << k;
        }
        EXPECT_LE(orthogonality_loss(r.vectors), 100.0 * n * eps) << which;
        for (idx k = 0; k < n; ++k) {
            vec<real> v(n);
            for (idx i = 0; i < n; ++i) {
                v[i] = r.vectors(i, k);
            }
            const vec<real> Wv = dense_times(W, v);
            real res = 0;
            for (idx i = 0; i < n; ++i) {
                res = std::max(res, std::abs(Wv[i] - r.values[k] * v[i]));
            }
            EXPECT_LE(res, 100.0 * n * eps * 12.0) << which << " k = " << k;
        }
    };
    check(eig_sym(assume_symmetric(W)), "default");
    check(seq::eig_sym(W, 1e-14, 100), "seq");
}

TEST(ScaleInvariance, EigensolversIgnoreTheScale) {
    constexpr idx n = 12;
    for (real s : {1e-20, 1e20}) {
        const mat<real> A = scaled_laplacian(n, s);
        const auto e = seq::eig_sym(A, 1e-12, 100);
        const auto e1 = seq::eig_sym(scaled_laplacian(n, 1.0), 1e-12, 100);
        EXPECT_TRUE(e.converged) << "s = " << s;
        EXPECT_EQ(e.sweeps, e1.sweeps) << "s = " << s;
        for (idx k = 0; k < n; ++k) {
            EXPECT_NEAR(e.values[k] / s, e1.values[k], 1e-12) << "s = " << s;
        }

        const auto l = lanczos(assume_symmetric(A), 2, 1e-8, n);
        EXPECT_TRUE(l.converged) << "s = " << s;
        EXPECT_NEAR(l.ritz_values[1] / s, e1.values[n - 1], 1e-8) << "s = " << s;
    }
}
