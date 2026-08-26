/// @file structures/graph/clique.hpp
/// @brief Exact and randomized clique reduction (star-mesh transformation) on multigraphs.
#pragma once

#include "structures/containers/degree_queue.hpp"
#include "structures/graph/multigraph.hpp"
#include <algorithm>
#include <concepts>
#include <cstdint>
#include <random>
#include <vector>

namespace num::structures {

template <typename Weight = double, std::integral Index = num::idx,
          typename Queue = structures::BasicDegreeQueue<Index>>
inline Weight collect_neighbors(Index v, std::vector<std::vector<MultiEdge<Weight, Index>>> &G,
                                const std::vector<std::uint8_t> &done, Queue &q,
                                std::vector<MultiEdge<Weight, Index>> &star_buf,
                                std::vector<MultiEdge<Weight, Index>> &nbr_out) {
    star_buf.clear();
    for (const auto &e : G[v]) {
        if (done[e.to]) {
            continue;
        }
        star_buf.push_back(e);
        q.rekey(e.to, q.degree_of(e.to) - e.count);
    }

    G[v].clear();
    G[v].shrink_to_fit();

    if (star_buf.empty()) {
        return static_cast<Weight>(0);
    }

    std::sort(star_buf.begin(), star_buf.end(),
              [](const MultiEdge<Weight, Index> &a, const MultiEdge<Weight, Index> &b) {
                  return a.to < b.to;
              });

    nbr_out.clear();
    Weight total_weight = static_cast<Weight>(0);

    for (const auto &e : star_buf) {
        if (!nbr_out.empty() && nbr_out.back().to == e.to) {
            nbr_out.back().weight += e.weight;
            nbr_out.back().count =
                static_cast<std::uint8_t>(std::min<Index>(255, nbr_out.back().count + e.count));
        } else {
            nbr_out.push_back(e);
        }
        total_weight += e.weight;
    }

    return total_weight;
}

template <typename Weight = double, std::integral Index = num::idx,
          typename Queue = structures::BasicDegreeQueue<Index>>
inline void add_exact_clique(std::vector<std::vector<MultiEdge<Weight, Index>>> &G, Queue &q,
                             const std::vector<MultiEdge<Weight, Index>> &nbr, Weight total_weight) {
    for (Index a = 0; a < nbr.size(); ++a) {
        for (Index b = a + 1; b < nbr.size(); ++b) {
            Weight w_exact = (nbr[a].weight * nbr[b].weight) / total_weight;
            Index u = nbr[a].to;
            Index j = nbr[b].to;

            G[u].push_back({j, w_exact, 1});
            G[j].push_back({u, w_exact, 1});
            q.rekey(u, q.degree_of(u) + 1);
            q.rekey(j, q.degree_of(j) + 1);
        }
    }
}

template <typename Weight = double, std::integral Index = num::idx,
          typename Queue = structures::BasicDegreeQueue<Index>, typename Rng = std::mt19937_64>
inline void sample_clique(std::vector<std::vector<MultiEdge<Weight, Index>>> &G, Queue &q,
                          std::vector<MultiEdge<Weight, Index>> &nbr, Weight total_weight,
                          std::type_identity_t<Index> sample_limit, Rng &rng) {
    std::sort(nbr.begin(), nbr.end(),
              [](const MultiEdge<Weight, Index> &a, const MultiEdge<Weight, Index> &b) {
                  return a.weight < b.weight;
              });

    Weight rest = total_weight;
    std::uniform_real_distribution<Weight> unit_dist(static_cast<Weight>(0), static_cast<Weight>(1));

    for (Index a = 0; a + 1 < nbr.size(); ++a) {
        const auto &nbr_a = nbr[a];
        rest -= nbr_a.weight;

        if (rest <= static_cast<Weight>(0)) {
            break;
        }

        const Index draws = std::min<Index>(nbr_a.count, sample_limit);
        const Weight w_bar = nbr_a.weight / static_cast<Weight>(draws);
        const Weight w_new = w_bar * (rest / total_weight);

        for (Index s = 0; s < draws; ++s) {
            Weight target = unit_dist(rng) * rest;

            Index b = a + 1;
            while (b + 1 < nbr.size() && target >= nbr[b].weight) {
                target -= nbr[b].weight;
                ++b;
            }

            Index u = nbr_a.to;
            Index j = nbr[b].to;

            G[u].push_back({j, w_new, 1});
            G[j].push_back({u, w_new, 1});
            q.rekey(u, q.degree_of(u) + 1);
            q.rekey(j, q.degree_of(j) + 1);
        }
    }
}

} // namespace num::structures
