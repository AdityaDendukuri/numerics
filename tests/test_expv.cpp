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
