/// @file linear/graph/randommat/approxchol.hpp
/// @brief Randomized Approximate Cholesky (ApproxChol) factorizations and SDD/Laplacian
/// preconditioners.
#pragma once

#include "linear/graph/randommat/solve.hpp"
#include "linear/graph/randommat/types.hpp"
#include "stochastic/rng.hpp"
#include "structures/containers/degree_queue.hpp"
#include "structures/graph/clique.hpp"
#include "structures/graph/graph.hpp"
#include "structures/graph/multigraph.hpp"
#include <cmath>
#include <concepts>
#include <cstdint>
#include <optional>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

namespace num {

namespace randommat {

namespace detail {

template <typename Float, std::integral Index>
inline void write_column(Index v, Index step, Float total_weight,
                         const std::vector<neighbor<Float, Index>> &nbr,
                         cholesky_factor<Float, Index> &factor) {
    const Float sqrt_total = std::sqrt(total_weight);

    auto &col_entries = factor.columns[step].entries;
    col_entries.reserve(nbr.size() + 1);
    col_entries.push_back({v, sqrt_total});
    for (const auto &n_info : nbr) {
        col_entries.push_back({n_info.to, -n_info.weight / sqrt_total});
    }
}

template <typename Float, std::integral Index>
inline void permute_rows(cholesky_factor<Float, Index> &factor) {
    const Index n = static_cast<Index>(factor.order.size());
    std::vector<Index> position(n);

    for (Index step = 0; step < n; ++step) {
        position[factor.order[step]] = step;
    }

    for (Index step = 0; step < n; ++step) {
        for (auto &entry : factor.columns[step].entries) {
            entry.row = position[entry.row];
        }
    }
}

} // namespace detail

/// @brief How the star-mesh clique left by each elimination is approximated.
enum class clique_sampler : std::uint8_t {
    exact,       ///< Keep the full clique. Correct, and densifies.
    independent, ///< Kyng-Sachdeva: `samples` independent draws per neighbour.
    tree,        ///< A random spanning tree of the clique, reweighted to stay unbiased.
};

/// Factorize a multigraph Laplacian into approximate or exact Cholesky factor \f$L L^T\f$.
template <typename Float = double, std::integral Index = num::idx, typename Rng = rng64,
          typename Queue = structures::basic_degree_queue<Index>>
inline cholesky_factor<Float, Index> factorize(const graph<Float, Index> &input_G,
                                               std::type_identity_t<Index> samples = 1,
                                               bool exact_mode = false, Rng *rng = nullptr,
                                               clique_sampler sampler = clique_sampler::independent,
                                               std::optional<Index> pinned_vertex = std::nullopt) {
    if (exact_mode) {
        sampler = clique_sampler::exact;
    }
    const Index n = static_cast<Index>(input_G.size());
    if (pinned_vertex && *pinned_vertex >= n) {
        throw std::out_of_range("factorize: pinned vertex is outside the graph");
    }
    graph<Float, Index> G = input_G;

    if (samples > 1 && !exact_mode) {
        for (auto &row : G) {
            for (auto &e : row) {
                e.count = static_cast<std::uint8_t>(samples);
            }
        }
    }

    Queue q(n);
    for (Index i = 0; i < n; ++i) {
        Index d = 0;
        for (const auto &e : G[i]) {
            d += e.count;
        }
        q.insert(i, d);
    }
    std::vector<std::uint8_t> done(n, 0);

    cholesky_factor<Float, Index> factor;
    factor.columns.resize(n);
    factor.order.reserve(n);

    std::vector<graph_edge<Float, Index>> star_buf;
    star_buf.reserve(128);

    std::vector<neighbor<Float, Index>> nbr_buf;
    nbr_buf.reserve(128);

    Rng local_rng(42);
    Rng &active_rng = rng ? *rng : local_rng;

    for (Index step = 0; step + 1 < n; ++step) {
        Index v = q.pop_min();
        if (pinned_vertex && v == *pinned_vertex) {
            const Index pinned_degree = q.degree_of(v);
            v = q.pop_min();
            q.insert(*pinned_vertex, pinned_degree);
        }
        factor.order.push_back(v);

        Float total_weight = structures::collect_neighbors(v, G, done, q, star_buf, nbr_buf);
        done[v] = 1;

        if (total_weight <= static_cast<Float>(0)) {
            continue;
        }

        detail::write_column(v, step, total_weight, nbr_buf, factor);

        switch (sampler) {
        case clique_sampler::exact:
            structures::add_exact_clique(G, q, nbr_buf, total_weight);
            break;
        case clique_sampler::tree:
            structures::sample_clique_tree(G, q, nbr_buf, total_weight, active_rng,
                                           static_cast<std::size_t>(samples));
            break;
        case clique_sampler::independent:
            structures::sample_clique(G, q, nbr_buf, total_weight, samples, active_rng);
            break;
        }
    }

    // Nullspace vertex
    const Index last_v = q.pop_min();
    if (pinned_vertex && last_v != *pinned_vertex) {
        throw std::runtime_error("factorize: failed to retain pinned vertex");
    }
    factor.order.push_back(last_v);

    detail::permute_rows(factor);

    return factor;
}

/// ApproxChol factorizer with 1 random sample per clique node.
/// @return `cholesky_factor`: the sparse approximate factor, consumed by `randommat::solve`.
template <typename Float = double, std::integral Index = num::idx>
inline cholesky_factor<Float, Index> ac1(const graph<Float, Index> &G, std::uint64_t seed = 42) {
    rng64 rng(seed);
    return factorize<Float, Index, rng64>(G, 1, false, &rng);
}

template <typename Float = double, std::integral Index = num::idx, typename Rng = rng64>
inline cholesky_factor<Float, Index> ac1(const graph<Float, Index> &G, Rng &rng) {
    return factorize<Float, Index, Rng>(G, 1, false, &rng);
}

/// ApproxChol factorizer with 2 random samples per clique node.
/// @return `cholesky_factor`: the sparse approximate factor, consumed by `randommat::solve`.
template <typename Float = double, std::integral Index = num::idx>
inline cholesky_factor<Float, Index> ac2(const graph<Float, Index> &G, std::uint64_t seed = 42) {
    rng64 rng(seed);
    return factorize<Float, Index, rng64>(G, 2, false, &rng);
}

template <typename Float = double, std::integral Index = num::idx, typename Rng = rng64>
inline cholesky_factor<Float, Index> ac2(const graph<Float, Index> &G, Rng &rng) {
    return factorize<Float, Index, Rng>(G, 2, false, &rng);
}

/// @brief Spanning-tree clique sampler: one reweighted random tree per elimination.
///
/// Keeps \f$d-1\f$ edges per eliminated vertex instead of \f$d\f$, drawn from a
/// negatively dependent distribution rather than independently. See
/// `num::structures::sample_clique_tree` for why the reweighting needs no
/// effective-resistance solve.
/// @return `cholesky_factor`: the sparse approximate factor, consumed by `randommat::solve`.
template <typename Float = double, std::integral Index = num::idx, typename Rng = rng64>
inline cholesky_factor<Float, Index> act(const graph<Float, Index> &G, Rng &rng,
                                         std::type_identity_t<Index> trees = 1) {
    return factorize<Float, Index, Rng>(G, trees, false, &rng, clique_sampler::tree);
}

/// Exact sparse Cholesky factorization via full star-mesh elimination.
/// @return `cholesky_factor`: the sparse approximate factor, consumed by `randommat::solve`.
template <typename Float = double, std::integral Index = num::idx>
inline cholesky_factor<Float, Index> exact(const graph<Float, Index> &G) {
    return factorize<Float, Index, rng64>(G, 1, true, nullptr);
}

} // namespace randommat
} // namespace num
