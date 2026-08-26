#include "container/matrix.hpp"
#include "container/matrix_ops.hpp"
#include "container/vector.hpp"
#include "linear/eigen/eigen.hpp"
#include "operator/operator.hpp"

#include <cmath>
#include <gtest/gtest.h>

using namespace num;

template <class Op>
concept LanczosCallable = requires(const Op &A) {
    lanczos(A, 2);
};

static_assert(!LanczosCallable<operators::DenseOp>);

// Helpers

static Matrix make_sym(idx n) {
    Matrix A(n, n, 0.0);
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

static real reconstruction_error(const Matrix &A, const EigenResult &r) {
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

static real orthogonality_error(const Matrix &V) {
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
    Matrix A(3, 3, 0.0);
    A(0, 0) = 4;
    A(0, 1) = 1;
    A(0, 2) = 0;
    A(1, 0) = 1;
    A(1, 1) = 3;
    A(1, 2) = 1;
    A(2, 0) = 0;
    A(2, 1) = 1;
    A(2, 2) = 2;
    auto r = eig_sym(assume_symmetric(A), 1e-12, 100, backend::seq);
    EXPECT_TRUE(r.converged);
    EXPECT_LT(reconstruction_error(A, r), 1e-10);
    EXPECT_LT(orthogonality_error(r.vectors), 1e-10);
}

TEST(EigSym_Jacobi, EigenvaluesAscending) {
    Matrix A = make_sym(8);
    auto r = eig_sym(assume_symmetric(A), 1e-12, 100, backend::seq);
    for (idx i = 1; i < r.values.size(); ++i) {
        EXPECT_LE(r.values[i - 1], r.values[i] + 1e-12);
    }
}

TEST(EigSym_Jacobi, ReconstructN32) {
    Matrix A = make_sym(32);
    auto r = eig_sym(assume_symmetric(A), 1e-12, 100, backend::seq);
    EXPECT_LT(reconstruction_error(A, r), 1e-8);
    EXPECT_LT(orthogonality_error(r.vectors), 1e-8);
}

// LAPACK

#if defined(NUMERICS_HAS_LAPACK)

TEST(EigSym_LAPACK, Reconstruct3x3) {
    Matrix A(3, 3, 0.0);
    A(0, 0) = 4;
    A(0, 1) = 1;
    A(0, 2) = 0;
    A(1, 0) = 1;
    A(1, 1) = 3;
    A(1, 2) = 1;
    A(2, 0) = 0;
    A(2, 1) = 1;
    A(2, 2) = 2;
    auto r = eig_sym(assume_symmetric(A), 1e-12, 100, backend::lapack);
    EXPECT_TRUE(r.converged);
    EXPECT_LT(reconstruction_error(A, r), 1e-10);
    EXPECT_LT(orthogonality_error(r.vectors), 1e-10);
}

TEST(EigSym_LAPACK, MatchesJacobi) {
    Matrix A = make_sym(20);
    auto rj = eig_sym(assume_symmetric(A), 1e-12, 100, backend::seq);
    auto rl = eig_sym(assume_symmetric(A), 1e-12, 100, backend::lapack);
    ASSERT_EQ(rj.values.size(), rl.values.size());
    for (idx i = 0; i < rj.values.size(); ++i) {
        EXPECT_NEAR(rj.values[i], rl.values[i], 1e-8);
    }
}

TEST(EigSym_LAPACK, ReconstructN64) {
    Matrix A = make_sym(64);
    auto r = eig_sym(assume_symmetric(A), 1e-12, 100, backend::lapack);
    EXPECT_LT(reconstruction_error(A, r), 1e-8);
    EXPECT_LT(orthogonality_error(r.vectors), 1e-8);
}

#endif // NUMERICS_HAS_LAPACK

// Power iteration

TEST(PowerIteration, DominantEigenvalue) {
    // Diagonal matrix: dominant eigenvalue = 10
    idx n = 5;
    Matrix A(n, n, 0.0);
    A(0, 0) = 10;
    A(1, 1) = 5;
    A(2, 2) = 3;
    A(3, 3) = 2;
    A(4, 4) = 1;
    auto r = power_iteration(A, 1e-10, 1000, backend::seq);
    EXPECT_TRUE(r.converged);
    EXPECT_NEAR(std::abs(r.eigenvalue), 10.0, 1e-8);
}

// Lanczos

TEST(Lanczos, TopKEigenvalues) {
    idx n = 50;
    Matrix A = make_sym(n);
    auto op =
        operators::make_op([&](const Vector &v, Vector &w) { matvec(A, v, w, backend::seq); }, n);
    auto r = lanczos(operators::assume_symmetric(op), 5, 1e-10);
    EXPECT_TRUE(r.converged);

    // Compare against Jacobi for top 5 eigenvalues
    auto ref = eig_sym(assume_symmetric(A), 1e-12, 100, backend::seq);
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
    Matrix A = make_sym(n);
    operators::DenseOp op(A, backend::seq);
    static_assert(SelfAdjointOperator<decltype(operators::assume_symmetric(op))>);
    auto r = lanczos(operators::assume_symmetric(op), 5, 1e-10);
    EXPECT_TRUE(r.converged);

    auto ref = eig_sym(assume_symmetric(A), 1e-12, 100, backend::seq);
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
