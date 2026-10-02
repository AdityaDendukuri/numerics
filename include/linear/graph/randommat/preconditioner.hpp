/// @file linear/graph/randommat/preconditioner.hpp
/// @brief Adapting the ApproxChol factorization to numerics' operator vocabulary.
///
/// `approxchol.hpp` needs only `num::structures` and can be lifted out alone. This header
/// converts a `num::spmat` Laplacian to its graph and wraps the factor as a
/// `num::preconditioner` for `num::pcg`.
#pragma once

#include "container/vector.hpp"
#include "linear/matrix_properties.hpp"
#include "linear/solvers/solver_result.hpp"
#include "operator/concepts.hpp"
#include "linear/graph/laplacian.hpp"
#include "linear/graph/randommat/approxchol.hpp"
#include "linear/sparse/sparse.hpp"
#include "stochastic/rng.hpp"
#include <cmath>
#include <stdexcept>
#include <utility>

namespace num {
namespace randommat {

/// @brief preconditioner adapter backed by Randomized Approximate Cholesky (ApproxChol).
/// Satisfies the num::preconditioner concept for use with num::pcg and Krylov solvers.
class approx_chol_preconditioner final {
  public:
    using domain_type = vec<real>;
    using codomain_type = vec<real>;
    // A graph-Laplacian factor is singular on the constant-vector nullspace.
    // PCG on a compatible subspace needs `law::spd_on<S>`, which the caller attaches.
    using laws = law::list<law::psd>;

    /// Construct from an existing factor.
    explicit approx_chol_preconditioner(cholesky_factor<real, idx> factor)
        : factor_(std::move(factor)), n_(factor_.order.size()), scratch_(n_, 0.0) {}

    /// Number of rows.
    [[nodiscard]] idx rows() const noexcept { return n_; }

    /// Number of columns.
    [[nodiscard]] idx cols() const noexcept { return n_; }

    /// Apply preconditioner z = M^-1 r via forward and backward substitution.
    void apply(const vec<real> &r, vec<real> &z) const {
        if (r.size() != n_) {
            throw std::invalid_argument("approx_chol_preconditioner: dimension mismatch");
        }
        if (z.size() != n_) {
            z = vec<real>(n_, 0.0);
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

/// Sparse approximate factor C of a grounded SDDM matrix, stored through the
/// Cholesky factor of its one-vertex Laplacian extension.  The ground vertex is
/// pinned last and omitted from all vector actions below.
class grounded_approx_chol_factor final {
  public:
    explicit grounded_approx_chol_factor(cholesky_factor<real, idx> factor)
        : factor_(std::move(factor)),
          n_(factor_.order.empty() ? 0 : static_cast<idx>(factor_.order.size() - 1)) {
        if (factor_.order.empty() || factor_.order.back() != n_) {
            throw std::invalid_argument(
                "grounded_approx_chol_factor: ground vertex must be ordered last");
        }
    }

    [[nodiscard]] idx rows() const noexcept { return n_; }
    [[nodiscard]] idx cols() const noexcept { return n_; }

    /// Apply C, where the approximate grounded matrix is C C^T.
    [[nodiscard]] vec<real> apply_lower(const vec<real> &x) const {
        check_dimension(x);
        const vec<real> permuted = permute(x);
        vec<real> product(n_, 0.0);
        for (idx column = 0; column < n_; ++column) {
            const auto &entries = retained_column(column);
            product[column] += entries[0].value * permuted[column];
            for (std::size_t entry = 1; entry < entries.size(); ++entry)
                if (entries[entry].row < n_)
                    product[entries[entry].row] += entries[entry].value * permuted[column];
        }
        return inverse_permute(product);
    }

    /// Apply C^{-1} by forward substitution.
    [[nodiscard]] vec<real> solve_lower(const vec<real> &b) const {
        check_dimension(b);
        vec<real> solution = permute(b);
        for (idx column = 0; column < n_; ++column) {
            const auto &entries = retained_column(column);
            solution[column] /= entries[0].value;
            const real value = solution[column];
            for (std::size_t entry = 1; entry < entries.size(); ++entry)
                if (entries[entry].row < n_)
                    solution[entries[entry].row] -= entries[entry].value * value;
        }
        return inverse_permute(solution);
    }

    /// Apply C^{-T} by backward substitution.
    [[nodiscard]] vec<real> solve_upper(const vec<real> &b) const {
        check_dimension(b);
        vec<real> solution = permute(b);
        for (idx step = 0; step < n_; ++step) {
            const idx column = n_ - 1 - step;
            const auto &entries = retained_column(column);
            real value = solution[column];
            for (std::size_t entry = 1; entry < entries.size(); ++entry)
                if (entries[entry].row < n_)
                    value -= entries[entry].value * solution[entries[entry].row];
            solution[column] = value / entries[0].value;
        }
        return inverse_permute(solution);
    }

    [[nodiscard]] const cholesky_factor<real, idx> &factor() const noexcept { return factor_; }

  private:
    void check_dimension(const vec<real> &x) const {
        if (x.size() != n_)
            throw std::invalid_argument("grounded_approx_chol_factor: dimension mismatch");
    }

    [[nodiscard]] const std::vector<factor_entry<real, idx>> &retained_column(idx column) const {
        const auto &entries = factor_.columns[column].entries;
        if (entries.empty())
            throw std::runtime_error("grounded_approx_chol_factor: singular retained factor");
        return entries;
    }

    [[nodiscard]] vec<real> permute(const vec<real> &x) const {
        vec<real> result(n_, 0.0);
        for (idx position = 0; position < n_; ++position)
            result[position] = x[factor_.order[position]];
        return result;
    }

    [[nodiscard]] vec<real> inverse_permute(const vec<real> &x) const {
        vec<real> result(n_, 0.0);
        for (idx position = 0; position < n_; ++position)
            result[factor_.order[position]] = x[position];
        return result;
    }

    cholesky_factor<real, idx> factor_;
    idx n_ = 0;
};

static_assert(linear_operator<approx_chol_preconditioner, vec<real>, vec<real>>);

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
template <detail::gks_2023_algorithm Algorithm>
[[nodiscard]] inline approx_chol_preconditioner
approxchol_preconditioner(const graph<real, idx> &G, Algorithm algorithm, std::uint64_t seed = 42) {
    rng64 rng(seed);
    auto factor = factorize<real, idx>(G, algorithm, &rng);
    return approx_chol_preconditioner(std::move(factor));
}

/// Construct ApproxChol preconditioner from a num::basic_graph.
template <typename Weight, std::integral Index, detail::gks_2023_algorithm Algorithm>
[[nodiscard]] inline approx_chol_preconditioner
approxchol_preconditioner(const basic_graph<Weight, Index> &G, Algorithm algorithm,
                          std::uint64_t seed = 42) {
    auto ac_G = to_approxchol_graph(G);
    return approxchol_preconditioner(ac_G, algorithm, seed);
}

/// Construct ApproxChol preconditioner from a num::multigraph.
template <typename Weight, std::integral Index, detail::gks_2023_algorithm Algorithm>
[[nodiscard]] inline approx_chol_preconditioner
approxchol_preconditioner(const structures::basic_multigraph<Weight, Index> &mg,
                          Algorithm algorithm, std::uint64_t seed = 42) {
    return approxchol_preconditioner(mg.adjacency(), algorithm, seed);
}

/// Construct ApproxChol preconditioner from a Laplacian spmat.
template <detail::gks_2023_algorithm Algorithm>
[[nodiscard]] inline approx_chol_preconditioner
approxchol_preconditioner(const spmat &L, Algorithm algorithm, std::uint64_t seed = 42) {
    auto ac_G = to_approxchol_graph(L);
    return approxchol_preconditioner(ac_G, algorithm, seed);
}

/// Construct a sparse approximate factor of a symmetric diagonally dominant
/// M-matrix by adding one ground vertex for the diagonal excess.
template <detail::gks_2023_algorithm Algorithm>
[[nodiscard]] inline grounded_approx_chol_factor
grounded_approxchol_factor(const spmat &A, Algorithm algorithm, std::uint64_t seed = 42) {
    const idx n = A.n_rows();
    if (A.n_cols() != n)
        throw std::invalid_argument("grounded_approxchol_factor: matrix must be square");

    structures::multigraph graph_with_ground(n + 1);
    bool has_ground_edge = false;
    for (idx row = 0; row < n; ++row) {
        real diagonal = 0.0;
        real off_diagonal_sum = 0.0;
        for (idx entry = A.row_ptr()[row]; entry < A.row_ptr()[row + 1]; ++entry) {
            const idx column = A.col_idx()[entry];
            const real value = A.values()[entry];
            if (column == row) {
                diagonal += value;
            } else {
                if (value > 1e-12)
                    throw std::invalid_argument(
                        "grounded_approxchol_factor: positive off-diagonal entry");
                off_diagonal_sum -= value;
                if (column > row && value < 0.0)
                    graph_with_ground.add_edge(row, column, -value, 1);
            }
        }
        real ground_weight = diagonal - off_diagonal_sum;
        const real tolerance = 1e-12 * std::max(real{1}, std::abs(diagonal));
        if (ground_weight < -tolerance)
            throw std::invalid_argument(
                "grounded_approxchol_factor: matrix is not diagonally dominant");
        if (ground_weight > tolerance) {
            graph_with_ground.add_edge(row, n, ground_weight, 1);
            has_ground_edge = true;
        }
    }
    if (!has_ground_edge)
        throw std::invalid_argument(
            "grounded_approxchol_factor: matrix has no positive grounding term");

    rng64 rng(seed);
    auto factor = factorize<real, idx>(graph_with_ground.adjacency(), algorithm, &rng, n);
    return grounded_approx_chol_factor(std::move(factor));
}

} // namespace randommat


// Convenience top-level num:: aliases
using approx_chol_preconditioner = randommat::approx_chol_preconditioner;
using grounded_approx_chol_factor = randommat::grounded_approx_chol_factor;
using randommat::approxchol_preconditioner;
using randommat::grounded_approxchol_factor;
using randommat::to_approxchol_graph;

namespace gao_kyng_spielman_2023 = randommat::gao_kyng_spielman_2023;

} // namespace num
