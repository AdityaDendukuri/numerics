/// @file structures/graph/clique.hpp
/// @brief Exact and randomized clique reduction (star-mesh transformation) on multigraphs.
#pragma once

#include "core/types.hpp"
#include "structures/containers/degree_queue.hpp"
#include "stochastic/rng.hpp"
#include "structures/graph/multigraph.hpp"
#include <algorithm>
#include <concepts>
#include <cstdint>
#include <random>
#include <vector>

namespace num::structures {

/// @brief Gather the neighbors of `v`, merging parallel edges, and return their total weight.
template <typename Weight = double, std::integral Index = num::idx,
          typename Queue = structures::basic_degree_queue<Index>>
inline Weight collect_neighbors(Index v, array<array<multi_edge<Weight, Index>>> &G,
                                const array<std::uint8_t> &done, Queue &q,
                                array<multi_edge<Weight, Index>> &star_buf,
                                array<multi_edge<Weight, Index>> &nbr_out) {
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
              [](const multi_edge<Weight, Index> &a, const multi_edge<Weight, Index> &b) {
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

/// @brief Add the exact clique that eliminating a vertex creates among its neighbors.
template <typename Weight = double, std::integral Index = num::idx,
          typename Queue = structures::basic_degree_queue<Index>>
inline void add_exact_clique(array<array<multi_edge<Weight, Index>>> &G, Queue &q,
                             const array<multi_edge<Weight, Index>> &nbr, Weight total_weight) {
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

/// @brief Add a sampled sparse approximation of that clique, with at most `sample_limit` draws
/// per neighbor.
template <typename Weight = double, std::integral Index = num::idx,
          typename Queue = structures::basic_degree_queue<Index>, typename Rng = rng64>
inline void sample_clique(array<array<multi_edge<Weight, Index>>> &G, Queue &q,
                          array<multi_edge<Weight, Index>> &nbr, Weight total_weight,
                          std::type_identity_t<Index> sample_limit, Rng &rng) {
    std::sort(nbr.begin(), nbr.end(),
              [](const multi_edge<Weight, Index> &a, const multi_edge<Weight, Index> &b) {
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


/// @brief Replace the star-mesh clique by a random spanning tree of it, reweighted
/// so the elimination stays unbiased in expectation.
///
/// Eliminating `v` adds a clique on its neighbours with weights \f$w_{ij} = c_i c_j / C\f$,
/// where \f$c_i\f$ is the conductance from `v` to `i` and \f$C = \sum_i c_i\f$. Edge
/// \f$ij\f$ enters the tree with probability \f$p_{ij} = (c_i + c_j)/C\f$ and is reweighted
/// to the harmonic mean \f$c_i c_j / (c_i + c_j)\f$, so \f$\mathbb{E}[\widetilde L^{(v)}] =
/// \mathrm{Sc}(L)\f$. The walk is capped: past the cap the remaining vertices attach to the
/// heaviest neighbour, which keeps a spanning tree but loses exactness for that elimination.
///
/// @param G Adjacency being eliminated; sampled edges are appended.
/// @param q Degree queue, rekeyed for each endpoint touched.
/// @param nbr Neighbours of the eliminated vertex, with their conductances.
/// @param total_weight \f$C\f$, the sum of those conductances.
/// @param rng Random source.
template <typename Weight, std::integral Index, typename Queue, typename Rng>
inline void sample_clique_tree(array<array<multi_edge<Weight, Index>>> &G, Queue &q,
                               const array<multi_edge<Weight, Index>> &nbr,
                               Weight total_weight, Rng &rng,
                               std::size_t trees = 1) {
    const std::size_t degree = nbr.size();
    if (degree < 2 || !(total_weight > Weight(0)) || trees == 0) {
        return;
    }

    array<Weight> cumulative(degree);
    Weight running = Weight(0);
    std::size_t heaviest = 0;
    for (std::size_t i = 0; i < degree; ++i) {
        running += nbr[i].weight;
        cumulative[i] = running;
        if (nbr[i].weight > nbr[heaviest].weight) {
            heaviest = i;
        }
    }
    if (!(running > Weight(0))) {
        return;
    }

    std::uniform_real_distribution<Weight> unit(Weight(0), Weight(1));
    const auto draw = [&]() -> std::size_t {
        const Weight target = unit(rng) * running;
        std::size_t low = 0;
        std::size_t high = degree - 1;
        while (low < high) {
            const std::size_t mid = low + ((high - low) / 2);
            if (cumulative[mid] < target) {
                low = mid + 1;
            } else {
                high = mid;
            }
        }
        return low;
    };

    const auto connect = [&](std::size_t a, std::size_t b) {
        const Weight wa = nbr[a].weight;
        const Weight wb = nbr[b].weight;
        const Weight sum = wa + wb;
        if (!(sum > Weight(0))) {
            return;
        }
        // Harmonic mean w_ij/p_ij, split across the trees so the average of
        // `trees` independent samples remains unbiased.
        const Weight weight = (wa * wb) / (sum * static_cast<Weight>(trees));
        const Index u = nbr[a].to;
        const Index v = nbr[b].to;
        G[u].push_back({v, weight, 1});
        G[v].push_back({u, weight, 1});
        q.rekey(u, q.degree_of(u) + 1);
        q.rekey(v, q.degree_of(v) + 1);
    };

    for (std::size_t tree = 0; tree < trees; ++tree) {
    array<char> visited(degree, 0);
    std::size_t current = draw();
    visited[current] = 1;
    std::size_t remaining = degree - 1;

    // Coupon-collector expectation is d*H_d for uniform weights; the cap leaves
    // generous room for moderate skew before the deterministic fallback.
    const std::size_t budget = (8 * degree * (1 + degree / 4)) + 64;
    for (std::size_t step = 0; step < budget && remaining > 0; ++step) {
        const std::size_t next = draw();
        if (next == current) {
            continue;
        }
        if (visited[next] == 0) {
            connect(current, next);
            visited[next] = 1;
            --remaining;
        }
        current = next;
    }
    if (remaining > 0) {
        for (std::size_t i = 0; i < degree; ++i) {
            if (visited[i] == 0 && i != heaviest) {
                connect(heaviest, i);
                visited[i] = 1;
            }
        }
    }
    }
}


/// @brief Sample a spanning tree of the directed product biclique left by an LU pivot.
///
/// Eliminating pivot `v` leaves the rank-one Schur update \f$xy^{T}/a\f$, with
/// \f$x_i = -A_{iv}\f$, \f$y_j = -A_{vj}\f$ and \f$a = A_{vv}\f$. Each in-neighbour becomes
/// \f$i_L\f$ and each out-neighbour \f$j_R\f$ of a bipartite graph with conductances
/// \f$x_iy_j/a\f$. Edge \f$i_L j_R\f$ enters the tree with probability
/// \f$p_{ij} = 1 - (1-\alpha_i)(1-\beta_j)\f$, where \f$\alpha_i = x_i/X\f$ and
/// \f$\beta_j = y_j/Y\f$, and reweighting by \f$w_{ij}/p_{ij}\f$ gives
/// \f$\mathbb{E}[\widetilde S_v] = xy^{T}/a\f$. It requires \f$x_i, y_j, a > 0\f$, as in a
/// nonsymmetric M-matrix; otherwise nothing is emitted.
///
/// @param x Incoming conductances, all strictly positive.
/// @param y Outgoing conductances, all strictly positive.
/// @param pivot \f$a = A_{vv}\f$, strictly positive.
/// @param rng Random source.
/// @param trees Independent trees to average; each sampled weight is divided by this.
/// @param emit Callable `void(std::size_t i, std::size_t j, Weight w)` receiving directed fill.
template <typename Weight, typename Rng, typename Emit>
inline void sample_biclique_tree(const array<Weight> &x, const array<Weight> &y,
                                 Weight pivot, Rng &rng, std::size_t trees, Emit emit) {
    const std::size_t m = x.size();
    const std::size_t n = y.size();
    if (m == 0 || n == 0 || trees == 0 || !(pivot > Weight(0))) {
        return;
    }

    Weight total_x = Weight(0);
    Weight total_y = Weight(0);
    for (const Weight value : x) {
        if (!(value > Weight(0))) {
            return; // signed entries leave the conductance reading
        }
        total_x += value;
    }
    for (const Weight value : y) {
        if (!(value > Weight(0))) {
            return;
        }
        total_y += value;
    }
    if (!(total_x > Weight(0)) || !(total_y > Weight(0))) {
        return;
    }

    array<Weight> cumulative_x(m);
    array<Weight> cumulative_y(n);
    Weight running = Weight(0);
    for (std::size_t i = 0; i < m; ++i) {
        running += x[i];
        cumulative_x[i] = running;
    }
    running = Weight(0);
    for (std::size_t j = 0; j < n; ++j) {
        running += y[j];
        cumulative_y[j] = running;
    }

    std::uniform_real_distribution<Weight> unit(Weight(0), Weight(1));
    const auto pick = [&](const array<Weight> &cumulative, Weight total) -> std::size_t {
        const Weight target = unit(rng) * total;
        std::size_t low = 0;
        std::size_t high = cumulative.size() - 1;
        while (low < high) {
            const std::size_t mid = low + ((high - low) / 2);
            if (cumulative[mid] < target) {
                low = mid + 1;
            } else {
                high = mid;
            }
        }
        return low;
    };

    const auto record = [&](std::size_t i, std::size_t j) {
        const Weight alpha = x[i] / total_x;
        const Weight beta = y[j] / total_y;
        const Weight inclusion = Weight(1) - ((Weight(1) - alpha) * (Weight(1) - beta));
        if (!(inclusion > Weight(0))) {
            return;
        }
        const Weight exact = (x[i] * y[j]) / pivot;
        emit(i, j, exact / (inclusion * static_cast<Weight>(trees)));
    };

    const std::size_t budget = (8 * (m + n) * (1 + ((m + n) / 4))) + 64;
    for (std::size_t tree = 0; tree < trees; ++tree) {
        array<char> seen_left(m, 0);
        array<char> seen_right(n, 0);
        std::size_t remaining = m + n - 1;

        std::size_t current = pick(cumulative_x, total_x);
        bool on_left = true;
        seen_left[current] = 1;

        for (std::size_t step = 0; step < budget && remaining > 0; ++step) {
            if (on_left) {
                const std::size_t next = pick(cumulative_y, total_y);
                if (seen_right[next] == 0) {
                    record(current, next);
                    seen_right[next] = 1;
                    --remaining;
                }
                current = next;
            } else {
                const std::size_t next = pick(cumulative_x, total_x);
                if (seen_left[next] == 0) {
                    record(next, current);
                    seen_left[next] = 1;
                    --remaining;
                }
                current = next;
            }
            on_left = !on_left;
        }

        if (remaining > 0) {
            // Budget exhausted under a very skewed weight distribution. Attach
            // the stragglers to the heaviest vertex on the opposite side, which
            // keeps the result a spanning tree at the cost of exactness here.
            std::size_t heavy_left = 0;
            std::size_t heavy_right = 0;
            for (std::size_t i = 1; i < m; ++i) {
                if (x[i] > x[heavy_left]) {
                    heavy_left = i;
                }
            }
            for (std::size_t j = 1; j < n; ++j) {
                if (y[j] > y[heavy_right]) {
                    heavy_right = j;
                }
            }
            if (seen_left[heavy_left] == 0) {
                record(heavy_left, heavy_right);
                seen_left[heavy_left] = 1;
            }
            if (seen_right[heavy_right] == 0) {
                record(heavy_left, heavy_right);
                seen_right[heavy_right] = 1;
            }
            for (std::size_t i = 0; i < m; ++i) {
                if (seen_left[i] == 0) {
                    record(i, heavy_right);
                    seen_left[i] = 1;
                }
            }
            for (std::size_t j = 0; j < n; ++j) {
                if (seen_right[j] == 0) {
                    record(heavy_left, j);
                    seen_right[j] = 1;
                }
            }
        }
    }
}

} // namespace num::structures
