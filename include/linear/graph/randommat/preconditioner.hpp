/// @file linear/graph/randommat/preconditioner.hpp
/// @brief Approximate Cholesky preconditioners for PCG.
#pragma once

#include "container/vector.hpp"
#include "linear/graph/laplacian.hpp"
#include "linear/graph/randommat/approxchol.hpp"
#include "linear/matrix_properties.hpp"
#include "linear/solvers/solver_result.hpp"
#include "linear/sparse/sparse.hpp"
#include "operator/concepts.hpp"
#include "stochastic/rng.hpp"
#include <cmath>
#include <stdexcept>
#include <utility>

namespace num {
namespace randommat {

/// @brief Preconditioner from an approximate Cholesky factor: \f$M^{-1} = C^{-T} C^{-1}\f$.
class approx_chol_preconditioner final {
  public:
    using domain_type = vec<real>;
    using codomain_type = vec<real>;
    // A graph-Laplacian factor is singular on the constant-vector nullspace.
    // PCG on a compatible subspace needs `law::spd_on<S>`, which the caller attaches.
    using laws = law::list<law::psd>;

    /// @brief Construct the preconditioner from an existing factor.
    explicit approx_chol_preconditioner(cholesky_factor<real, idx> factor)
        : factor_(std::move(factor)), n_(factor_.order.size()), scratch_(n_, 0.0) {}

    /// @brief Return the number of rows.
    [[nodiscard]] idx rows() const noexcept { return n_; }

    /// @brief Return the number of columns.
    [[nodiscard]] idx cols() const noexcept { return n_; }

    /// @brief Apply the preconditioner: \f$z \leftarrow C^{-T} C^{-1} r\f$.
    void apply(const vec<real> &r, vec<real> &z) const {
        if (r.size() != n_) {
            throw std::invalid_argument("approx_chol_preconditioner: dimension mismatch");
        }
        if (z.size() != n_) {
            z = vec<real>(n_, 0.0);
        }
        randommat::solve(factor_, r.data(), z.data(), scratch_);
    }

    /// @brief Return the factor.
    [[nodiscard]] const cholesky_factor<real, idx> &factor() const noexcept { return factor_; }

  private:
    cholesky_factor<real, idx> factor_;
    idx n_ = 0;
    mutable std::vector<real> scratch_;
};

/// @brief Approximate Cholesky factor of an SDDM matrix, through an extra ground vertex:
/// \f$A \approx C C^T\f$.
class grounded_approx_chol_factor final {
  public:
    using domain_type = vec<real>;
    using codomain_type = vec<real>;
    using laws = law::list<law::spd>;

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

    /// @brief Multiply by the factor: \f$y \leftarrow C x\f$.
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

    /// @brief Solve with the factor by forward substitution: \f$y \leftarrow C^{-1} b\f$.
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

    /// @brief Solve with the transposed factor by backward substitution:
    /// \f$y \leftarrow C^{-T} b\f$.
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

    /// @brief Apply the preconditioner: \f$x \leftarrow C^{-T} C^{-1} b\f$.
    void apply(const vec<real> &b, vec<real> &x) const { x = solve_upper(solve_lower(b)); }

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
static_assert(math::spd_operator<grounded_approx_chol_factor, vec<real>>);

/// @brief Convert a graph to adjacency lists.
template <typename Weight, std::integral Index>
[[nodiscard]] inline adjacency_list<real, idx> to_approxchol_graph(const graph<Weight, Index> &G) {
    const auto mg = structures::to_multigraph(G);
    return mg.adjacency();
}

/// @brief Convert a multigraph to adjacency lists.
template <typename Weight, std::integral Index>
[[nodiscard]] inline adjacency_list<real, idx>
to_approxchol_graph(const structures::multigraph<Weight, Index> &mg) {
    return mg.adjacency();
}

/// @brief Convert a Laplacian to adjacency lists.
[[nodiscard]] inline adjacency_list<real, idx> to_approxchol_graph(const spmat &L) {
    const auto mg = num::linear::to_multigraph(L);
    return mg.adjacency();
}

/// @brief Construct an approximate Cholesky preconditioner from adjacency lists.
template <detail::gks_2023_algorithm Algorithm>
[[nodiscard]] inline approx_chol_preconditioner
approxchol_preconditioner(const adjacency_list<real, idx> &G, Algorithm algorithm,
                          std::uint64_t seed = 42) {
    rng64 rng(seed);
    auto factor = factorize<real, idx>(G, algorithm, &rng);
    return approx_chol_preconditioner(std::move(factor));
}

/// @brief Construct an approximate Cholesky preconditioner from a graph.
template <typename Weight, std::integral Index, detail::gks_2023_algorithm Algorithm>
[[nodiscard]] inline approx_chol_preconditioner
approxchol_preconditioner(const graph<Weight, Index> &G, Algorithm algorithm,
                          std::uint64_t seed = 42) {
    auto ac_G = to_approxchol_graph(G);
    return approxchol_preconditioner(ac_G, algorithm, seed);
}

/// @brief Construct an approximate Cholesky preconditioner from a multigraph.
template <typename Weight, std::integral Index, detail::gks_2023_algorithm Algorithm>
[[nodiscard]] inline approx_chol_preconditioner
approxchol_preconditioner(const structures::multigraph<Weight, Index> &mg, Algorithm algorithm,
                          std::uint64_t seed = 42) {
    return approxchol_preconditioner(mg.adjacency(), algorithm, seed);
}

/// @brief Construct an approximate Cholesky preconditioner from a Laplacian.
template <detail::gks_2023_algorithm Algorithm>
[[nodiscard]] inline approx_chol_preconditioner
approxchol_preconditioner(const spmat &L, Algorithm algorithm, std::uint64_t seed = 42) {
    auto ac_G = to_approxchol_graph(L);
    return approxchol_preconditioner(ac_G, algorithm, seed);
}

/// @brief Compute an approximate Cholesky factor of an SDDM matrix via a ground vertex:
/// \f$A \approx C C^T\f$.
template <detail::gks_2023_algorithm Algorithm>
[[nodiscard]] inline grounded_approx_chol_factor
grounded_approxchol_factor(const spmat &A, Algorithm algorithm, std::uint64_t seed = 42) {
    const idx n = A.n_rows();
    if (A.n_cols() != n)
        throw std::invalid_argument("grounded_approxchol_factor: matrix must be square");

    structures::multigraph<real, idx> graph_with_ground(n + 1);
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
