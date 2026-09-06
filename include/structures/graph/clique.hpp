/// @file structures/graph/clique.hpp
/// @brief Exact and randomized clique reduction (star-mesh transformation) on multigraphs.
#pragma once

#include "core/types.hpp"
#include "structures/containers/degree_queue.hpp"
#include "structures/graph/multigraph.hpp"
#include <algorithm>
#include <concepts>
#include <cstdint>
#include <random>
#include <vector>

namespace num::structures {

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

template <typename Weight = double, std::integral Index = num::idx,
          typename Queue = structures::basic_degree_queue<Index>, typename Rng = std::mt19937_64>
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
/// Eliminating `v` adds a clique on its neighbours with the Schur-complement
/// weights \f$w_{ij} = c_i c_j / C\f$, where \f$c_i\f$ is the conductance from
/// `v` to neighbour `i` and \f$C = \sum_i c_i\f$. `sample_clique` approximates
/// that clique by independent draws; this samples a *spanning tree* of it
/// instead, keeping \f$d-1\f$ edges rather than \f$d\f$ and drawing them from a
/// negatively dependent (strongly Rayleigh) distribution.
///
/// ### Why this is unbiased, and why it needs no effective-resistance solve
///
/// For a weighted uniform spanning tree, Kirchhoff gives the inclusion
/// probability \f$p_e = w_e R_{\mathrm{eff}}(e)\f$. Computing
/// \f$R_{\mathrm{eff}}\f$ in general needs Laplacian solves — the very problem
/// this factorization exists to precondition — but on *this* clique it is
/// closed form. The clique Laplacian is a rank-one update of a diagonal,
/// \f$\operatorname{diag}(c) - cc^{T}/C\f$, and solving it gives
/// \f[
///   R_{\mathrm{eff}}(i,j) = \frac{1}{c_i} + \frac{1}{c_j},
/// \f]
/// which is just the series path \f$i \to v \to j\f$ that the clique replaced.
/// Hence
/// \f[
///   p_{ij} = \frac{c_i + c_j}{C}, \qquad
///   \sum_{i<j} p_{ij} = d - 1,
/// \f]
/// exactly the edge count of a spanning tree, and the reweighting
/// \f$\widehat c_{ij} = w_{ij}/p_{ij} = c_i c_j / (c_i + c_j)\f$ is the harmonic
/// mean — the same series conductance the independent sampler assigns. So
/// \f$\mathbb{E}[\widetilde L^{(v)}] = \mathrm{Sc}(L)\f$ holds by construction.
///
/// ### Aldous--Broder degenerates here
///
/// The clique's transition kernel is \f$P(i \to j) = c_j / (C - c_i)\f$, which
/// does not depend on where the walk currently is. Sampling the tree is
/// therefore not a graph traversal but a coupon-collector process: draw
/// i.i.d. from \f$c/C\f$, and record the entering edge on each first visit.
///
/// ### Measured behaviour
///
/// On 2D grid Laplacians (n up to 40k), one tree per elimination is *worse* than
/// the independent sampler: 44 PCG iterations against `ac1`'s 41 and `ac2`'s 29,
/// and on higher-degree graphs with skewed weights the coupon-collector walk
/// makes setup several times more expensive as well. Unbiasedness alone does not
/// buy a preconditioner -- a single tree has too much variance.
///
/// Averaging `trees` independent trees per elimination does concentrate, as the
/// strongly-Rayleigh matrix Chernoff theory predicts, and overtakes the existing
/// samplers on quality (n = 14400, grid): 44 iterations at k=1, 27 at k=2, 20 at
/// k=4, 15 at k=8, against `ac2`'s 29. Setup cost grows faster than k, because
/// each tree adds edges that raise the degree of every later elimination:
/// 11 ms, 82 ms, 616 ms, 2.7 s for the same k. So this pays only when one
/// factorization is reused across many solves -- roughly 50 solves to break even
/// at k=2, 600 at k=8 -- and loses outright for a single solve.
///
/// A skewed weight distribution also makes the collection slow, so the walk is
/// capped. On reaching the cap the remaining vertices are attached to the
/// heaviest neighbour, which keeps the result a spanning tree and the cost
/// bounded, at the price of exactness in expectation for that elimination.
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
/// Eliminating pivot `v` from a nonsymmetric matrix leaves the rank-one Schur
/// update \f$xy^{T}/a\f$, with \f$x_i = -A_{iv}\f$, \f$y_j = -A_{vj}\f$ and
/// \f$a = A_{vv}\f$. That is not an undirected clique: \f$i \to j\f$ and
/// \f$j \to i\f$ carry genuinely different weights \f$x_iy_j/a\f$ and
/// \f$x_jy_i/a\f$.
///
/// Lifting to an undirected *bipartite* graph on duplicated vertices recovers the
/// symmetric machinery without losing that asymmetry: put \f$i_L\f$ for each
/// incoming neighbour, \f$j_R\f$ for each outgoing one, and give edge
/// \f$i_L - j_R\f$ conductance \f$x_iy_j/a\f$. Opposite directed edges become
/// \f$i_L-j_R\f$ and \f$j_L-i_R\f$, distinct objects, so nothing is forced
/// together. A vertex that is both an in- and out-neighbour contributes
/// \f$i_L-i_R\f$, which maps back to the diagonal fill \f$x_iy_i/a\f$ — the lift
/// handles diagonal and off-diagonal fill in one object.
///
/// ### Closed-form marginals again
///
/// Writing \f$X = \sum_i x_i\f$, \f$Y = \sum_j y_j\f$, solving the bipartite
/// Laplacian gives
/// \f[
///   R_{\mathrm{eff}}(i_L, j_R)
///     = a\Big[\tfrac{1}{x_iY} + \tfrac{1}{y_jX} - \tfrac{1}{XY}\Big],
/// \f]
/// so with \f$\alpha_i = x_i/X\f$ and \f$\beta_j = y_j/Y\f$,
/// \f[
///   p_{ij} = w_{ij}R_{\mathrm{eff}} = \alpha_i + \beta_j - \alpha_i\beta_j
///          = 1 - (1-\alpha_i)(1-\beta_j) \in [0,1],
/// \f]
/// which sums to \f$m + n - 1\f$: the edge count of a spanning tree on the
/// \f$m+n\f$ duplicated vertices. Reweighting by \f$w_{ij}/p_{ij}\f$ therefore
/// leaves \f$\mathbb{E}[\widetilde S_v] = xy^{T}/a\f$, the exact LU Schur update.
///
/// The walk simplifies too: \f$P(i_L \to j_R) = y_j/Y\f$ and
/// \f$P(j_R \to i_L) = x_i/X\f$, neither depending on where it currently is, so
/// tree sampling is a bipartite coupon collector alternating two fixed
/// distributions.
///
/// ### Restriction
///
/// The conductance reading needs \f$x_i, y_j, a > 0\f$ — a nonsymmetric M-matrix
/// or directed Laplacian. Under general signed LU the "conductances" go negative
/// and none of the above applies. Callers are responsible for that check; this
/// routine returns without emitting anything if it is violated.
///
/// Note also that unbiasedness is weaker here than in the symmetric case. mat
/// Chernoff bounds for strongly Rayleigh measures are Hermitian results, so they
/// say nothing about \f$\|\widetilde S_v - S_v\|\f$ concentrating for a
/// nonsymmetric update. Whether this makes a good preconditioner is an empirical
/// question, not one the expectation settles.
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
