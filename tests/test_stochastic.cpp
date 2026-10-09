#include "stochastic/categorical.hpp"
#include "stochastic/multinomial.hpp"
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
    const mat<real> probe = rademacher_probe(5, 8, generator);
    EXPECT_EQ(probe.rows(), 5);
    EXPECT_EQ(probe.cols(), 8);
    for (idx j = 0; j < probe.rows(); ++j)
        for (idx p = 0; p < probe.cols(); ++p)
            EXPECT_DOUBLE_EQ(probe(j, p) * probe(j, p), 1.0);
    EXPECT_THROW(rademacher_probe(5, 0, generator), std::invalid_argument);
}

TEST(probe, SeededOverloadIsReproducible) {
    const mat<real> first = rademacher_probe(6, 4, 11u);
    const mat<real> second = rademacher_probe(6, 4, 11u);
    for (idx j = 0; j < 6; ++j)
        for (idx p = 0; p < 4; ++p)
            EXPECT_DOUBLE_EQ(first(j, p), second(j, p));
}

TEST(probe, HutchinsonRecoversDiagonalOfDiagonalMatrix) {
    // B = diag(d): probed = B z has rows d_j z_j, so the mean square is d_j^2 exactly.
    const vec<real> d{1.0, -2.0, 0.5};
    const mat<real> probe = rademacher_probe(3, 16, 5u);
    mat<real> probed(3, 16, 0.0);
    for (idx j = 0; j < 3; ++j)
        for (idx p = 0; p < 16; ++p)
            probed(j, p) = d[j] * probe(j, p);
    const vec<real> estimate = hutchinson_row_mean_square(probed);
    for (idx j = 0; j < 3; ++j)
        EXPECT_NEAR(estimate[j], d[j] * d[j], 1e-14);
    EXPECT_THROW(hutchinson_row_mean_square(mat<real>(3, 0, 0.0)), std::invalid_argument);
}

TEST(multinomial, CountsSumToTheRequestedNumberOfDraws) {
    rng generator(13);
    const vec<real> weights{1.0, 2.0, 3.0, 4.0};
    const array<idx> count = sample_multinomial(weights, 1000, generator);
    EXPECT_EQ(count.size(), weights.size());
    EXPECT_EQ(count[0] + count[1] + count[2] + count[3], 1000);
}

TEST(multinomial, HandlesUnnormalizedAndDegenerateWeights) {
    rng generator(17);
    const vec<real> certain{0.0, 7.0, 0.0};
    EXPECT_EQ(sample_multinomial(certain, 23, generator), (array<idx>{0, 23, 0}));
    EXPECT_EQ(sample_multinomial(certain, 0, generator), (array<idx>{0, 0, 0}));

    const vec<real> negative{1.0, -1.0};
    const vec<real> zero{0.0, 0.0};
    const vec<real> empty;
    EXPECT_THROW(sample_multinomial(negative, 1, generator), std::invalid_argument);
    EXPECT_THROW(sample_multinomial(zero, 1, generator), std::invalid_argument);
    EXPECT_THROW(sample_multinomial(empty, 1, generator), std::invalid_argument);
}
