#include "stochastic/categorical.hpp"
#include "stochastic/probe.hpp"
#include "stochastic/rng.hpp"
#include <gtest/gtest.h>
#include <type_traits>

using namespace num;

TEST(rng, AliasesAreTheStandardEngines) {
    static_assert(std::is_same_v<rng, std::mt19937>);
    static_assert(std::is_same_v<rng64, std::mt19937_64>);
    rng a = markov::make_rng(7);
    rng b(7);
    EXPECT_EQ(a(), b());
}

TEST(probe, RademacherEntriesAreSigns) {
    rng generator(3);
    const mat probe = rademacher_probe(5, 8, generator);
    EXPECT_EQ(probe.rows(), 5);
    EXPECT_EQ(probe.cols(), 8);
    for (idx j = 0; j < probe.rows(); ++j)
        for (idx p = 0; p < probe.cols(); ++p)
            EXPECT_DOUBLE_EQ(probe(j, p) * probe(j, p), 1.0);
    EXPECT_THROW(rademacher_probe(5, 0, generator), std::invalid_argument);
}

TEST(probe, SeededOverloadIsReproducible) {
    const mat first = rademacher_probe(6, 4, 11u);
    const mat second = rademacher_probe(6, 4, 11u);
    for (idx j = 0; j < 6; ++j)
        for (idx p = 0; p < 4; ++p)
            EXPECT_DOUBLE_EQ(first(j, p), second(j, p));
}

TEST(probe, HutchinsonRecoversDiagonalOfDiagonalMatrix) {
    // B = diag(d): probed = B z has rows d_j z_j, so the mean square is d_j^2 exactly.
    const vec d{1.0, -2.0, 0.5};
    const mat probe = rademacher_probe(3, 16, 5u);
    mat probed(3, 16, 0.0);
    for (idx j = 0; j < 3; ++j)
        for (idx p = 0; p < 16; ++p)
            probed(j, p) = d[j] * probe(j, p);
    const vec estimate = hutchinson_row_mean_square(probed);
    for (idx j = 0; j < 3; ++j)
        EXPECT_NEAR(estimate[j], d[j] * d[j], 1e-14);
    EXPECT_THROW(hutchinson_row_mean_square(mat(3, 0, 0.0)), std::invalid_argument);
}
