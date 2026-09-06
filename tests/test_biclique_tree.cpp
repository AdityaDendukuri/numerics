/// @file tests/test_biclique_tree.cpp
/// @brief Bipartite-lift tree sampler for the nonsymmetric (LU) Schur update.
///
/// Eliminating an LU pivot leaves the rank-one update \f$xy^{T}/a\f$, a directed
/// product biclique rather than an undirected clique. Lifting it to a bipartite
/// graph on duplicated vertices recovers the tree sampler, and the inclusion
/// probability again has a closed form,
/// \f[
///   p_{ij} = 1 - (1-\alpha_i)(1-\beta_j),
///   \qquad \alpha_i = x_i/X,\ \beta_j = y_j/Y.
/// \f]
///
/// The tests check the three factors of \f$p_{ij}\widehat w_{ij} = w_{ij}\f$
/// separately, then the whole estimator against the exact rank-one matrix. The
/// last test is the one that is not settled by any theory: matrix Chernoff
/// bounds for strongly Rayleigh measures are Hermitian results, so nothing
/// guarantees a nonsymmetric update concentrates. It is measured instead.

#include "structures/graph/clique.hpp"
#include <cmath>
#include <gtest/gtest.h>
#include <random>
#include <vector>

namespace {

struct Accumulated {
    std::vector<std::vector<double>> weight; // running sum of emitted weights
    std::vector<std::vector<long>> hits;
    long edges = 0;
};

Accumulated sample(const std::vector<double> &x, const std::vector<double> &y, double pivot,
                   int trials, std::size_t trees = 1, std::uint64_t seed = 20260831) {
    Accumulated out;
    out.weight.assign(x.size(), std::vector<double>(y.size(), 0.0));
    out.hits.assign(x.size(), std::vector<long>(y.size(), 0));
    std::mt19937_64 rng(seed);
    for (int t = 0; t < trials; ++t) {
        num::structures::sample_biclique_tree(x, y, pivot, rng, trees,
                                              [&](std::size_t i, std::size_t j, double w) {
                                                  out.weight[i][j] += w;
                                                  out.hits[i][j] += 1;
                                                  out.edges += 1;
                                              });
    }
    return out;
}

const std::vector<double> incoming{0.4, 1.3, 2.0, 0.7};
const std::vector<double> outgoing{1.1, 0.5, 2.4};
constexpr double pivot_value = 3.0;

double sum_of(const std::vector<double> &v) {
    double total = 0.0;
    for (const double value : v) {
        total += value;
    }
    return total;
}

TEST(BicliqueTree, KeepsExactlyOneSpanningTreeOfTheLiftedGraph) {
    const int trials = 3000;
    const auto result = sample(incoming, outgoing, pivot_value, trials);
    const double expected = static_cast<double>(incoming.size() + outgoing.size()) - 1.0;
    EXPECT_DOUBLE_EQ(static_cast<double>(result.edges) / trials, expected)
        << "a spanning tree on m+n duplicated vertices has m+n-1 edges";
}

TEST(BicliqueTree, InclusionProbabilitiesMatchTheClosedForm) {
    const double total_x = sum_of(incoming);
    const double total_y = sum_of(outgoing);
    const int trials = 300000;
    const auto result = sample(incoming, outgoing, pivot_value, trials);

    double predicted_total = 0.0;
    for (std::size_t i = 0; i < incoming.size(); ++i) {
        for (std::size_t j = 0; j < outgoing.size(); ++j) {
            const double alpha = incoming[i] / total_x;
            const double beta = outgoing[j] / total_y;
            const double predicted = 1.0 - ((1.0 - alpha) * (1.0 - beta));
            predicted_total += predicted;
            const double observed = static_cast<double>(result.hits[i][j]) / trials;
            EXPECT_NEAR(observed, predicted, 0.02 * predicted) << "edge (" << i << "," << j << ")";
        }
    }
    EXPECT_NEAR(predicted_total, static_cast<double>(incoming.size() + outgoing.size()) - 1.0, 1e-12)
        << "the marginals must sum to the tree's edge count";
}

TEST(BicliqueTree, InclusionProbabilityIsAlwaysAValidProbability) {
    // p = 1 - (1-a)(1-b) with a,b in (0,1] can never leave [0,1], however
    // lopsided the weights. A formula that could exceed 1 would make the
    // reweighting shrink an edge below its true weight.
    std::mt19937_64 rng(5);
    std::lognormal_distribution<double> weight(0.0, 3.0);
    for (int trial = 0; trial < 200; ++trial) {
        std::vector<double> x(4);
        std::vector<double> y(3);
        for (double &value : x) {
            value = weight(rng);
        }
        for (double &value : y) {
            value = weight(rng);
        }
        const double total_x = sum_of(x);
        const double total_y = sum_of(y);
        for (std::size_t i = 0; i < x.size(); ++i) {
            for (std::size_t j = 0; j < y.size(); ++j) {
                const double p =
                    1.0 - ((1.0 - (x[i] / total_x)) * (1.0 - (y[j] / total_y)));
                EXPECT_GE(p, 0.0);
                EXPECT_LE(p, 1.0);
            }
        }
    }
}

TEST(BicliqueTree, EstimatorIsUnbiasedForTheRankOneUpdate) {
    // The invariant: E[S~] = x y^T / a, entry by entry.
    const int trials = 300000;
    const auto result = sample(incoming, outgoing, pivot_value, trials);
    for (std::size_t i = 0; i < incoming.size(); ++i) {
        for (std::size_t j = 0; j < outgoing.size(); ++j) {
            const double expectation = result.weight[i][j] / trials;
            const double exact = (incoming[i] * outgoing[j]) / pivot_value;
            EXPECT_NEAR(expectation, exact, 0.03 * exact) << "entry (" << i << "," << j << ")";
        }
    }
}

TEST(BicliqueTree, AsymmetryIsPreserved) {
    // With x and y swapped the estimator must target the transpose, not the
    // same matrix: forcing opposite directed edges together would lose this.
    const std::vector<double> x{2.0, 0.5};
    const std::vector<double> y{0.25, 4.0};
    const int trials = 200000;
    const auto forward = sample(x, y, pivot_value, trials);
    const auto reversed = sample(y, x, pivot_value, trials, 1, 987654321);
    for (std::size_t i = 0; i < x.size(); ++i) {
        for (std::size_t j = 0; j < y.size(); ++j) {
            const double f = forward.weight[i][j] / trials;
            const double r = reversed.weight[j][i] / trials;
            EXPECT_NEAR(f, (x[i] * y[j]) / pivot_value, 0.03 * (x[i] * y[j]) / pivot_value);
            EXPECT_NEAR(r, (y[j] * x[i]) / pivot_value, 0.03 * (y[j] * x[i]) / pivot_value);
        }
    }
    // And the two directions genuinely differ.
    EXPECT_GT(std::abs((x[0] * y[1]) - (x[1] * y[0])), 1.0);
}

TEST(BicliqueTree, AveragingTreesKeepsTheExpectationAndConcentrates) {
    // Unbiasedness must survive averaging, and the deviation of a single draw
    // from the exact update must shrink with k. The second half is the part no
    // Hermitian concentration result covers -- matrix Chernoff bounds for
    // strongly Rayleigh measures do not apply to a nonsymmetric update -- so it
    // is measured rather than assumed.
    const int trials = 4000;
    const std::size_t rows = incoming.size();
    const std::size_t cols = outgoing.size();

    double previous_deviation = std::numeric_limits<double>::max();
    for (std::size_t trees : {std::size_t{1}, std::size_t{2}, std::size_t{4}, std::size_t{8},
                              std::size_t{16}}) {
        std::mt19937_64 rng(4242);
        double deviation_sum = 0.0;
        std::vector<std::vector<double>> mean(rows, std::vector<double>(cols, 0.0));

        for (int t = 0; t < trials; ++t) {
            std::vector<std::vector<double>> draw(rows, std::vector<double>(cols, 0.0));
            num::structures::sample_biclique_tree(
                incoming, outgoing, pivot_value, rng, trees,
                [&](std::size_t i, std::size_t j, double w) { draw[i][j] += w; });

            double squared = 0.0;
            for (std::size_t i = 0; i < rows; ++i) {
                for (std::size_t j = 0; j < cols; ++j) {
                    const double exact = (incoming[i] * outgoing[j]) / pivot_value;
                    const double error = draw[i][j] - exact;
                    squared += error * error;
                    mean[i][j] += draw[i][j];
                }
            }
            deviation_sum += std::sqrt(squared);
        }

        // Still unbiased at every k.
        for (std::size_t i = 0; i < rows; ++i) {
            for (std::size_t j = 0; j < cols; ++j) {
                const double exact = (incoming[i] * outgoing[j]) / pivot_value;
                EXPECT_NEAR(mean[i][j] / trials, exact, 0.08 * exact)
                    << trees << " trees, entry (" << i << "," << j << ")";
            }
        }

        const double deviation = deviation_sum / trials;
        EXPECT_LT(deviation, previous_deviation)
            << "k = " << trees << " did not reduce ||S~ - S||_F";
        previous_deviation = deviation;
    }
}

TEST(BicliqueTree, DeclinesToActOnSignedOrDegenerateInput) {
    // Outside the M-matrix regime the conductance reading is meaningless; the
    // sampler must emit nothing rather than produce a plausible-looking tree.
    std::mt19937_64 rng(3);
    int emitted = 0;
    const auto count = [&](std::size_t, std::size_t, double) { ++emitted; };

    num::structures::sample_biclique_tree(std::vector<double>{1.0, -2.0},
                                          std::vector<double>{1.0}, 1.0, rng, 1, count);
    EXPECT_EQ(emitted, 0) << "a negative incoming weight must be refused";

    num::structures::sample_biclique_tree(std::vector<double>{1.0},
                                          std::vector<double>{0.0}, 1.0, rng, 1, count);
    EXPECT_EQ(emitted, 0) << "a zero outgoing weight must be refused";

    num::structures::sample_biclique_tree(std::vector<double>{1.0}, std::vector<double>{1.0},
                                          -1.0, rng, 1, count);
    EXPECT_EQ(emitted, 0) << "a non-positive pivot must be refused";

    num::structures::sample_biclique_tree(std::vector<double>{}, std::vector<double>{1.0}, 1.0,
                                          rng, 1, count);
    EXPECT_EQ(emitted, 0) << "an empty neighbourhood has no biclique";
}

TEST(BicliqueTree, HandlesTheDiagonalFillEntry) {
    // A vertex that is both an in- and out-neighbour contributes i_L - i_R,
    // which maps back to the diagonal Schur entry x_i y_i / a.
    const std::vector<double> x{1.0, 2.0};
    const std::vector<double> y{3.0, 0.5};
    const int trials = 100000;
    const auto result = sample(x, y, 2.0, trials);
    for (std::size_t i = 0; i < 2; ++i) {
        const double expectation = result.weight[i][i] / trials;
        EXPECT_NEAR(expectation, (x[i] * y[i]) / 2.0, 0.04 * (x[i] * y[i]) / 2.0)
            << "diagonal entry " << i;
    }
}

} // namespace
