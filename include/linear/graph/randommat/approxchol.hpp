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
                         const std::vector<Neighbor<Float, Index>> &nbr,
                         CholeskyFactor<Float, Index> &factor) {
    const Float sqrt_total = std::sqrt(total_weight);

    auto &col_entries = factor.columns[step].entries;
    col_entries.reserve(nbr.size() + 1);
    col_entries.push_back({v, sqrt_total});
    for (const auto &n_info : nbr) {
        col_entries.push_back({n_info.to, -n_info.weight / sqrt_total});
    }
}

template <typename Float, std::integral Index>
inline void permute_rows(CholeskyFactor<Float, Index> &factor) {
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

/// Factorize a multigraph Laplacian into approximate or exact Cholesky factor \f$L L^T\f$.
template <typename Float = double, std::integral Index = num::idx, typename Rng = std::mt19937_64,
          typename Queue = structures::BasicDegreeQueue<Index>>
inline CholeskyFactor<Float, Index> factorize(const Graph<Float, Index> &input_G,
                                              std::type_identity_t<Index> samples = 1,
                                              bool exact_mode = false, Rng *rng = nullptr) {
    const Index n = static_cast<Index>(input_G.size());
    Graph<Float, Index> G = input_G;

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

    CholeskyFactor<Float, Index> factor;
    factor.columns.resize(n);
    factor.order.reserve(n);

    std::vector<Edge<Float, Index>> star_buf;
    star_buf.reserve(128);

    std::vector<Neighbor<Float, Index>> nbr_buf;
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

        if (exact_mode) {
            structures::add_exact_clique(G, q, nbr_buf, total_weight);
        } else {
            structures::sample_clique(G, q, nbr_buf, total_weight, samples, active_rng);
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
inline CholeskyFactor<Float, Index> ac1(const Graph<Float, Index> &G, std::uint64_t seed = 42) {
    std::mt19937_64 rng(seed);
    return factorize<Float, Index, std::mt19937_64>(G, 1, false, &rng);
}

template <typename Float = double, std::integral Index = num::idx, typename Rng = std::mt19937_64>
inline CholeskyFactor<Float, Index> ac1(const Graph<Float, Index> &G, Rng &rng) {
    return factorize<Float, Index, Rng>(G, 1, false, &rng);
}

/// ApproxChol factorizer with 2 random samples per clique node.
template <typename Float = double, std::integral Index = num::idx>
inline CholeskyFactor<Float, Index> ac2(const Graph<Float, Index> &G, std::uint64_t seed = 42) {
    std::mt19937_64 rng(seed);
    return factorize<Float, Index, std::mt19937_64>(G, 2, false, &rng);
}

template <typename Float = double, std::integral Index = num::idx, typename Rng = std::mt19937_64>
inline CholeskyFactor<Float, Index> ac2(const Graph<Float, Index> &G, Rng &rng) {
    return factorize<Float, Index, Rng>(G, 2, false, &rng);
}

/// Exact sparse Cholesky factorization via full star-mesh elimination.
template <typename Float = double, std::integral Index = num::idx>
inline CholeskyFactor<Float, Index> exact(const Graph<Float, Index> &G) {
    return factorize<Float, Index, std::mt19937_64>(G, 1, true, nullptr);
}

/// @brief Preconditioner adapter backed by Randomized Approximate Cholesky (ApproxChol).
/// Satisfies the num::Preconditioner concept for use with num::pcg and Krylov solvers.
class ApproxCholPreconditioner final {
  public:
    using domain_type = Vector;
    using codomain_type = Vector;
    // A graph-Laplacian factor is singular on the constant-vector nullspace.
    // Callers using PCG on a compatible subspace must attach that stronger,
    // problem-specific evidence explicitly.
    using math_propositions = math::type_list<axiom::positive_semidefinite>;

    /// Construct from an existing factor.
    explicit ApproxCholPreconditioner(CholeskyFactor<real, idx> factor)
        : factor_(std::move(factor)), n_(factor_.order.size()), scratch_(n_, 0.0) {}

    /// Number of rows.
    [[nodiscard]] idx rows() const noexcept { return n_; }

    /// Number of columns.
    [[nodiscard]] idx cols() const noexcept { return n_; }

    /// Apply preconditioner z = M^-1 r via forward and backward substitution.
    void apply(const Vector &r, Vector &z) const {
        if (r.size() != n_) {
            throw std::invalid_argument("ApproxCholPreconditioner: dimension mismatch");
        }
        if (z.size() != n_) {
            z = Vector(n_, 0.0);
        }
        randommat::solve(factor_, r.data(), z.data(), scratch_);
    }

    /// Access underlying factor.
    [[nodiscard]] const CholeskyFactor<real, idx> &factor() const noexcept { return factor_; }

  private:
    CholeskyFactor<real, idx> factor_;
    idx n_ = 0;
    mutable std::vector<real> scratch_;
};

static_assert(Preconditioner<ApproxCholPreconditioner>,
              "ApproxCholPreconditioner must satisfy num::Preconditioner concept");

/// Convert num::BasicGraph to randommat::Graph.
template <typename Weight, std::integral Index>
[[nodiscard]] inline Graph<real, idx> to_approxchol_graph(const BasicGraph<Weight, Index> &G) {
    const auto mg = structures::to_multigraph(G);
    return mg.adjacency();
}

/// Convert num::BasicMultigraph to randommat::Graph.
template <typename Weight, std::integral Index>
[[nodiscard]] inline Graph<real, idx>
to_approxchol_graph(const structures::BasicMultigraph<Weight, Index> &mg) {
    return mg.adjacency();
}

/// Convert Laplacian SparseMatrix (CSR) to randommat::Graph.
[[nodiscard]] inline Graph<real, idx> to_approxchol_graph(const SparseMatrix &L) {
    const auto mg = num::linear::to_multigraph(L);
    return mg.adjacency();
}

/// Construct ApproxChol preconditioner from a randommat::Graph.
[[nodiscard]] inline ApproxCholPreconditioner
approxchol_preconditioner(const Graph<real, idx> &G, idx samples = 1, std::uint64_t seed = 42) {
    std::mt19937_64 rng(seed);
    auto factor = factorize<real, idx>(G, samples, false, &rng);
    return ApproxCholPreconditioner(std::move(factor));
}

/// Construct ApproxChol preconditioner from a num::BasicGraph.
template <typename Weight, std::integral Index>
[[nodiscard]] inline ApproxCholPreconditioner
approxchol_preconditioner(const BasicGraph<Weight, Index> &G, idx samples = 1,
                          std::uint64_t seed = 42) {
    auto ac_G = to_approxchol_graph(G);
    return approxchol_preconditioner(ac_G, samples, seed);
}

/// Construct ApproxChol preconditioner from a num::Multigraph.
template <typename Weight, std::integral Index>
[[nodiscard]] inline ApproxCholPreconditioner
approxchol_preconditioner(const structures::BasicMultigraph<Weight, Index> &mg, idx samples = 1,
                          std::uint64_t seed = 42) {
    return approxchol_preconditioner(mg.adjacency(), samples, seed);
}

/// Construct ApproxChol preconditioner from a Laplacian SparseMatrix.
[[nodiscard]] inline ApproxCholPreconditioner
approxchol_preconditioner(const SparseMatrix &L, idx samples = 1, std::uint64_t seed = 42) {
    auto ac_G = to_approxchol_graph(L);
    return approxchol_preconditioner(ac_G, samples, seed);
}

} // namespace randommat

namespace math {

template <>
struct model_of<randommat::ApproxCholPreconditioner> {
    using laws = type_list<law::linear_map>;
};

} // namespace math

// Convenience top-level num:: aliases
using ApproxCholPreconditioner = randommat::ApproxCholPreconditioner;
using randommat::approxchol_preconditioner;
using randommat::to_approxchol_graph;

} // namespace num
