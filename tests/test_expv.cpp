#include <gtest/gtest.h>
#include <numerics.hpp>

using namespace num;

TEST(Expv, RestartedKrylovHandlesAStiffMarkovChain) {
    spmat Q(2, 2, {-1.0e6, 1.0, 1.0e6, -1.0}, {0, 1, 0, 1}, {0, 2, 4});
    vec<real> initial{1.0, 0.0};

    vec<real> probability =
        restarted_expv(1.0, Q, initial,
                       {.krylov_dimension = 2, .tolerance = 1e-9, .maximum_steps = 10000});
    real stationary_first = 1.0 / 1'000'001.0;

    EXPECT_NEAR(probability[0], stationary_first, 1e-8);
    EXPECT_NEAR(probability[0] + probability[1], 1.0, 1e-8);
}

TEST(Expv, RestartedKrylovAgreesWithOneShotOnASmallProblem) {
    mat<real> A(5, 5, 0.0);
    for (idx i = 0; i < 5; ++i)
        A(i, i) = -static_cast<real>(i + 1);
    vec<real> initial(5, 1.0);
    operators::dense_op op(A);

    vec<real> reference = expv(2.0, op, initial, 5, 1e-14);
    vec<real> restarted = restarted_expv(
        2.0, op, initial, {.krylov_dimension = 4, .tolerance = 1e-8});

    for (idx i = 0; i < reference.size(); ++i)
        EXPECT_NEAR(restarted[i], reference[i], 1e-7);
}

TEST(Expm, StiffGradedTriangularMatchesClosedForm) {
    // exp([a b; 0 c]) = [e^a, b (e^a - e^c)/(a - c); 0, e^c], with widely separated scales
    const real a = -2000.0, b = 1e3, c = -1e-3;
    mat<real> A(2, 2, 0.0);
    A(0, 0) = a;
    A(0, 1) = b;
    A(1, 1) = c;
    const mat<real> E = expm(A);
    const real e01 = b * (std::exp(a) - std::exp(c)) / (a - c);
    EXPECT_NEAR(E(1, 1), std::exp(c), 1e-13);
    EXPECT_NEAR(E(0, 1) / e01, 1.0, 1e-13);
    EXPECT_NEAR(E(1, 0), 0.0, 1e-300);
}

TEST(Expm, RotatedDiagonalMatchesEigendecomposition) {
    // A = Q diag(-30, -1, 0.5) Q^T with a fixed rotation Q
    const real ct = std::cos(0.7), st = std::sin(0.7);
    mat<real> Q(3, 3, 0.0);
    Q(0, 0) = ct;
    Q(0, 1) = -st;
    Q(1, 0) = st;
    Q(1, 1) = ct;
    Q(2, 2) = 1.0;
    const real lam[3] = {-30.0, -1.0, 0.5};
    mat<real> A(3, 3, 0.0), expected(3, 3, 0.0);
    for (idx i = 0; i < 3; ++i) {
        for (idx j = 0; j < 3; ++j) {
            for (idx k = 0; k < 3; ++k) {
                A(i, j) += Q(i, k) * lam[k] * Q(j, k);
                expected(i, j) += Q(i, k) * std::exp(lam[k]) * Q(j, k);
            }
        }
    }
    const mat<real> E = expm(A);
    for (idx i = 0; i < 3; ++i) {
        for (idx j = 0; j < 3; ++j) {
            EXPECT_NEAR(E(i, j), expected(i, j), 1e-14);
        }
    }
}
