/// @file tests/test_clique_tree.cpp
/// @brief Spanning-tree clique sampler and its unbiasedness invariant.
///
/// The whole construction rests on one identity. Eliminating a vertex leaves a
/// clique with Schur weights \f$w_{ij} = c_ic_j/C\f$; the sampler keeps a random
/// spanning tree of it and reweights by the inverse inclusion probability. For
/// the elimination to be correct in expectation,
/// \f[
///   \mathbb{E}[\widetilde L^{(v)}] = \mathrm{Sc}(L)
///   \quad\Longleftrightarrow\quad
///   p_{ij}\,\widehat c_{ij} = w_{ij},
/// \f]
/// and this file checks each factor of that separately: the inclusion
/// probability against its closed form, the reweighting against the harmonic
/// mean, and the product against the exact Schur weight, by sampling.
///
/// A sampler that produced a *connected* tree with the wrong marginals would
/// still solve linear systems, just with a preconditioner that quietly drifts
/// from the matrix it claims to approximate. Only the marginals catch that.

#include "structures/containers/degree_queue.hpp"
#include "structures/graph/clique.hpp"
#include <gtest/gtest.h>
#include <map>
#include <random>
#include <utility>
#include <vector>

namespace {

using graph_edge = num::structures::multi_edge<double, num::idx>;

struct Sampled {
    std::map<std::pair<num::idx, num::idx>, long> hits;
    std::map<std::pair<num::idx, num::idx>, double> weight;
    long edges = 0;
};

/// Run the sampler `trials` times over neighbour conductances `c`.
Sampled run(const std::vector<double> &c, int trials, std::size_t trees = 1) {
    const std::size_t degree = c.size();
    std::vector<graph_edge> nbr;
    double total = 0.0;
    for (std::size_t i = 0; i < degree; ++i) {
        nbr.push_back({static_cast<num::idx>(i), c[i], 1});
        total += c[i];
    }

    Sampled out;
    std::mt19937_64 rng(20260831);
    for (int t = 0; t < trials; ++t) {
        std::vector<std::vector<graph_edge>> graph(degree);
        num::structures::basic_degree_queue<num::idx> queue(degree);
        for (num::idx i = 0; i < degree; ++i) {
            queue.insert(i, 1);
        }
        num::structures::sample_clique_tree(graph, queue, nbr, total, rng, trees);
        for (num::idx u = 0; u < degree; ++u) {
            for (const auto &e : graph[u]) {
                if (u < e.to) {
                    out.hits[{u, e.to}] += 1;
                    out.weight[{u, e.to}] += e.weight;
                    out.edges += 1;
                }
            }
        }
    }
    return out;
}

TEST(CliqueTree, KeepsExactlyDegreeMinusOneEdges) {
    const std::vector<double> c{0.3, 1.0, 2.5, 0.7, 4.0, 1.5};
    const auto sampled = run(c, 2000);
    EXPECT_DOUBLE_EQ(static_cast<double>(sampled.edges) / 2000.0,
                     static_cast<double>(c.size()) - 1.0)
        << "a spanning tree of d neighbours has exactly d-1 edges";
}

TEST(CliqueTree, InclusionProbabilityMatchesItsClosedForm) {
    // p_ij = (c_i + c_j)/C, from R_eff(i,j) = 1/c_i + 1/c_j on the star-mesh
    // clique. This is what removes the need for an effective-resistance solve.
    const std::vector<double> c{0.3, 1.0, 2.5, 0.7, 4.0, 1.5};
    double total = 0.0;
    for (const double x : c) {
        total += x;
    }
    const int trials = 200000;
    const auto sampled = run(c, trials);

    double predicted_sum = 0.0;
    for (std::size_t i = 0; i < c.size(); ++i) {
        for (std::size_t j = i + 1; j < c.size(); ++j) {
            const double predicted = (c[i] + c[j]) / total;
            const double observed =
                static_cast<double>(
                    sampled.hits.at({static_cast<num::idx>(i), static_cast<num::idx>(j)})) /
                trials;
            predicted_sum += predicted;
            EXPECT_NEAR(observed, predicted, 0.02 * predicted)
                << "edge (" << i << "," << j << ")";
        }
    }
    EXPECT_NEAR(predicted_sum, static_cast<double>(c.size()) - 1.0, 1e-12)
        << "inclusion probabilities must sum to the tree's edge count";
}

TEST(CliqueTree, SampledWeightIsTheHarmonicMean) {
    const std::vector<double> c{0.5, 2.0, 3.0, 1.0};
    const auto sampled = run(c, 5000);
    for (std::size_t i = 0; i < c.size(); ++i) {
        for (std::size_t j = i + 1; j < c.size(); ++j) {
            const auto key = std::make_pair(static_cast<num::idx>(i), static_cast<num::idx>(j));
            const long count = sampled.hits.count(key) != 0 ? sampled.hits.at(key) : 0;
            ASSERT_GT(count, 0) << "edge (" << i << "," << j << ") never sampled";
            const double mean = sampled.weight.at(key) / static_cast<double>(count);
            EXPECT_NEAR(mean, (c[i] * c[j]) / (c[i] + c[j]), 1e-12)
                << "edge (" << i << "," << j << ")";
        }
    }
}

TEST(CliqueTree, EliminationIsUnbiasedInExpectation) {
    // The invariant itself: E[w sampled] must equal the exact Schur weight
    // c_i c_j / C, for every edge of the clique.
    const std::vector<double> c{0.4, 1.2, 2.0, 0.9, 3.0};
    double total = 0.0;
    for (const double x : c) {
        total += x;
    }
    const int trials = 200000;
    const auto sampled = run(c, trials);

    for (std::size_t i = 0; i < c.size(); ++i) {
        for (std::size_t j = i + 1; j < c.size(); ++j) {
            const auto key = std::make_pair(static_cast<num::idx>(i), static_cast<num::idx>(j));
            const double expectation = sampled.weight.at(key) / trials;
            const double exact = (c[i] * c[j]) / total;
            EXPECT_NEAR(expectation, exact, 0.03 * exact) << "edge (" << i << "," << j << ")";
        }
    }
}

TEST(CliqueTree, AveragingSeveralTreesPreservesTheExpectation) {
    // k trees each carry 1/k of the weight, so the estimator stays unbiased
    // while its variance falls. Only the second half is a performance claim;
    // the first is a correctness one.
    const std::vector<double> c{0.4, 1.2, 2.0, 0.9, 3.0};
    double total = 0.0;
    for (const double x : c) {
        total += x;
    }
    const int trials = 60000;
    for (std::size_t trees : {std::size_t{1}, std::size_t{2}, std::size_t{4}}) {
        const auto sampled = run(c, trials, trees);
        EXPECT_DOUBLE_EQ(static_cast<double>(sampled.edges) / trials,
                         static_cast<double>(trees) * (static_cast<double>(c.size()) - 1.0));
        for (std::size_t i = 0; i < c.size(); ++i) {
            for (std::size_t j = i + 1; j < c.size(); ++j) {
                const auto key = std::make_pair(static_cast<num::idx>(i), static_cast<num::idx>(j));
                const double expectation = sampled.weight.at(key) / trials;
                EXPECT_NEAR(expectation, (c[i] * c[j]) / total, 0.05 * (c[i] * c[j]) / total)
                    << trees << " trees, edge (" << i << "," << j << ")";
            }
        }
    }
}

TEST(CliqueTree, HandlesDegenerateNeighbourhoods) {
    std::vector<std::vector<graph_edge>> graph(2);
    num::structures::basic_degree_queue<num::idx> queue(2);
    queue.insert(0, 1);
    queue.insert(1, 1);
    std::mt19937_64 rng(1);

    // A single neighbour has no clique to sample.
    const std::vector<graph_edge> one{{0, 1.0, 1}};
    num::structures::sample_clique_tree(graph, queue, one, 1.0, rng);
    EXPECT_TRUE(graph[0].empty());

    // Zero total conductance must not divide by it.
    const std::vector<graph_edge> none{{0, 0.0, 1}, {1, 0.0, 1}};
    num::structures::sample_clique_tree(graph, queue, none, 0.0, rng);
    EXPECT_TRUE(graph[0].empty());
}

} // namespace
