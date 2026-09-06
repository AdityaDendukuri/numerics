/// @file linear/graph/randommat/approxchol.hpp
/// @brief Randomized Approximate Cholesky (ApproxChol) factorizations and SDD/Laplacian
/// preconditioners.
#pragma once

#include "container/vector.hpp"
#include "core/math/evidence.hpp"
#include "linear/concepts.hpp"
#include "linear/graph/laplacian.hpp"
#include "linear/graph/randommat/solve.hpp"
#include "linear/graph/randommat/types.hpp"
#include "linear/sparse/sparse.hpp"
#include "structures/containers/degree_queue.hpp"
#include "structures/graph/clique.hpp"
#include "structures/graph/graph.hpp"
#include "structures/graph/multigraph.hpp"
#include <cmath>
#include <concepts>
#include <cstdint>
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
template <typename Float = double, std::integral Index = num::idx, typename Rng = std::mt19937_64,
          typename Queue = structures::basic_degree_queue<Index>>
inline cholesky_factor<Float, Index> factorize(const graph<Float, Index> &input_G,
                                              std::type_identity_t<Index> samples = 1,
                                              bool exact_mode = false, Rng *rng = nullptr,
                                              clique_sampler sampler = clique_sampler::independent) {
    if (exact_mode) {
        sampler = clique_sampler::exact;
    }
    const Index n = static_cast<Index>(input_G.size());
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
        const Index v = q.pop_min();
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
    factor.order.push_back(last_v);

    detail::permute_rows(factor);

    return factor;
}

/// ApproxChol factorizer with 1 random sample per clique node.
template <typename Float = double, std::integral Index = num::idx>
inline cholesky_factor<Float, Index> ac1(const graph<Float, Index> &G, std::uint64_t seed = 42) {
    std::mt19937_64 rng(seed);
    return factorize<Float, Index, std::mt19937_64>(G, 1, false, &rng);
}

template <typename Float = double, std::integral Index = num::idx, typename Rng = std::mt19937_64>
inline cholesky_factor<Float, Index> ac1(const graph<Float, Index> &G, Rng &rng) {
    return factorize<Float, Index, Rng>(G, 1, false, &rng);
}

/// ApproxChol factorizer with 2 random samples per clique node.
template <typename Float = double, std::integral Index = num::idx>
inline cholesky_factor<Float, Index> ac2(const graph<Float, Index> &G, std::uint64_t seed = 42) {
    std::mt19937_64 rng(seed);
    return factorize<Float, Index, std::mt19937_64>(G, 2, false, &rng);
}

template <typename Float = double, std::integral Index = num::idx, typename Rng = std::mt19937_64>
inline cholesky_factor<Float, Index> ac2(const graph<Float, Index> &G, Rng &rng) {
    return factorize<Float, Index, Rng>(G, 2, false, &rng);
}

/// @brief Spanning-tree clique sampler: one reweighted random tree per elimination.
///
/// Keeps \f$d-1\f$ edges per eliminated vertex instead of \f$d\f$, drawn from a
/// negatively dependent distribution rather than independently. See
/// `num::structures::sample_clique_tree` for why the reweighting needs no
/// effective-resistance solve.
template <typename Float = double, std::integral Index = num::idx, typename Rng = std::mt19937_64>
inline cholesky_factor<Float, Index> act(const graph<Float, Index> &G, Rng &rng,
                                        std::type_identity_t<Index> trees = 1) {
    return factorize<Float, Index, Rng>(G, trees, false, &rng, clique_sampler::tree);
}

/// Exact sparse Cholesky factorization via full star-mesh elimination.
template <typename Float = double, std::integral Index = num::idx>
inline cholesky_factor<Float, Index> exact(const graph<Float, Index> &G) {
    return factorize<Float, Index, std::mt19937_64>(G, 1, true, nullptr);
}

/// @brief preconditioner adapter backed by Randomized Approximate Cholesky (ApproxChol).
/// Satisfies the num::preconditioner concept for use with num::pcg and Krylov solvers.
class approx_chol_preconditioner final {
  public:
    using domain_type = vec;
    using codomain_type = vec;
    // A graph-Laplacian factor is singular on the constant-vector nullspace.
    // Callers using PCG on a compatible subspace must attach that stronger,
    // problem-specific evidence explicitly.
    using math_laws = math::type_list<law::psd>;

    /// Construct from an existing factor.
    explicit approx_chol_preconditioner(cholesky_factor<real, idx> factor)
        : factor_(std::move(factor)), n_(factor_.order.size()), scratch_(n_, 0.0) {}

    /// Number of rows.
    [[nodiscard]] idx rows() const noexcept { return n_; }

    /// Number of columns.
    [[nodiscard]] idx cols() const noexcept { return n_; }

    /// Apply preconditioner z = M^-1 r via forward and backward substitution.
    void apply(const vec &r, vec &z) const {
        if (r.size() != n_) {
            throw std::invalid_argument("approx_chol_preconditioner: dimension mismatch");
        }
        if (z.size() != n_) {
            z = vec(n_, 0.0);
        }
        randommat::solve(factor_, r.data(), z.data(), scratch_);
    }

    /// Access underlying factor.
    [[nodiscard]] const cholesky_factor<real, idx> &factor() const noexcept { return factor_; }

  private:
    cholesky_factor<real, idx> factor_;
    idx n_ = 0;
    mutable std::vector<real> scratch_;
};

static_assert(preconditioner<approx_chol_preconditioner>,
              "approx_chol_preconditioner must satisfy num::preconditioner concept");

/// Convert num::basic_graph to randommat::graph.
template <typename Weight, std::integral Index>
[[nodiscard]] inline graph<real, idx> to_approxchol_graph(const basic_graph<Weight, Index> &G) {
    const auto mg = structures::to_multigraph(G);
    return mg.adjacency();
}

/// Convert num::basic_multigraph to randommat::graph.
template <typename Weight, std::integral Index>
[[nodiscard]] inline graph<real, idx>
to_approxchol_graph(const structures::basic_multigraph<Weight, Index> &mg) {
    return mg.adjacency();
}

/// Convert Laplacian spmat (CSR) to randommat::graph.
[[nodiscard]] inline graph<real, idx> to_approxchol_graph(const spmat &L) {
    const auto mg = num::linear::to_multigraph(L);
    return mg.adjacency();
}

/// Construct ApproxChol preconditioner from a randommat::graph.
[[nodiscard]] inline approx_chol_preconditioner
approxchol_preconditioner(const graph<real, idx> &G, idx samples = 1, std::uint64_t seed = 42) {
    std::mt19937_64 rng(seed);
    auto factor = factorize<real, idx>(G, samples, false, &rng);
    return approx_chol_preconditioner(std::move(factor));
}

/// Construct ApproxChol preconditioner from a num::basic_graph.
template <typename Weight, std::integral Index>
[[nodiscard]] inline approx_chol_preconditioner
approxchol_preconditioner(const basic_graph<Weight, Index> &G, idx samples = 1,
                          std::uint64_t seed = 42) {
    auto ac_G = to_approxchol_graph(G);
    return approxchol_preconditioner(ac_G, samples, seed);
}

/// Construct ApproxChol preconditioner from a num::multigraph.
template <typename Weight, std::integral Index>
[[nodiscard]] inline approx_chol_preconditioner
approxchol_preconditioner(const structures::basic_multigraph<Weight, Index> &mg, idx samples = 1,
                          std::uint64_t seed = 42) {
    return approxchol_preconditioner(mg.adjacency(), samples, seed);
}

/// Construct ApproxChol preconditioner from a Laplacian spmat.
[[nodiscard]] inline approx_chol_preconditioner
approxchol_preconditioner(const spmat &L, idx samples = 1, std::uint64_t seed = 42) {
    auto ac_G = to_approxchol_graph(L);
    return approxchol_preconditioner(ac_G, samples, seed);
}

} // namespace randommat

namespace math {

template <>
struct claims_of<randommat::approx_chol_preconditioner> {
    using type = type_list<law::linear_map>;
};

} // namespace math

// Convenience top-level num:: aliases
using approx_chol_preconditioner = randommat::approx_chol_preconditioner;
using randommat::approxchol_preconditioner;
using randommat::to_approxchol_graph;

} // namespace num
