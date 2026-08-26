/// @file linear/graph/laplacian.hpp
/// @brief Matrices produced from a graph.
///
/// A graph in `structures` carries no algebraic operations. The adjacency matrix,
/// the Laplacian, and the Markov generator are matrices, so they are produced
/// here, where matrix storage already lives.
///
/// The unnormalized Laplacian of an undirected weighted graph is
/// \f$L = D - W\f$, symmetric positive semi-definite with \f$L\mathbf{1} = 0\f$.
/// It is singular by construction, so tag it `assume_psd` rather than `assume_spd`.
#pragma once

#include "container/matrix.hpp"
#include "core/types.hpp"
#include "linear/sparse/sparse.hpp"
#include "structures/graph/graph.hpp"
#include "structures/graph/multigraph.hpp"

namespace num::linear {

/// @brief Graph from which a Laplacian can be assembled.
///
/// The Laplacian is symmetric positive semi-definite with \f$L\mathbf{1} = 0\f$,
/// and its null space has one dimension per connected component. Algorithms rely
/// on that, which is why the contract names the matrix rather than a method.
template <typename G>
concept LaplacianGraph = concepts::IncidenceStructure<G> && requires(const G &g) {
    {laplacian(g)};
};

template <typename Weight = real, typename Index = idx>
[[nodiscard]] inline SparseMatrix to_sparse_adjacency(const BasicGraph<Weight, Index> &g) {
    std::vector<idx> row_ptr(g.n_vertices() + 1, 0);
    std::vector<idx> col_idx;
    std::vector<double> values;

    for (Index u = 0; u < g.n_vertices(); ++u) {
        for (const auto &e : g.neighbors(u)) {
            col_idx.push_back(static_cast<idx>(e.to));
            values.push_back(static_cast<double>(e.weight));
        }
        row_ptr[u + 1] = col_idx.size();
    }
    return SparseMatrix(static_cast<idx>(g.n_vertices()), static_cast<idx>(g.n_vertices()),
                        std::move(values), std::move(col_idx), std::move(row_ptr));
}

template <typename Weight = real, typename Index = idx>
[[nodiscard]] inline Matrix to_dense_adjacency(const BasicGraph<Weight, Index> &g) {
    Matrix A(static_cast<idx>(g.n_vertices()), static_cast<idx>(g.n_vertices()), 0.0);
    for (Index u = 0; u < g.n_vertices(); ++u) {
        for (const auto &e : g.neighbors(u)) {
            A(static_cast<idx>(u), static_cast<idx>(e.to)) += static_cast<double>(e.weight);
        }
    }
    return A;
}

template <typename Weight = real, typename Index = idx>
[[nodiscard]] inline SparseMatrix laplacian(const BasicGraph<Weight, Index> &g) {
    std::vector<idx> rows;
    std::vector<idx> cols;
    std::vector<double> vals;

    for (Index u = 0; u < g.n_vertices(); ++u) {
        double deg = 0.0;
        for (const auto &e : g.neighbors(u)) {
            if (e.to != u) {
                rows.push_back(static_cast<idx>(u));
                cols.push_back(static_cast<idx>(e.to));
                vals.push_back(-static_cast<double>(e.weight));
                deg += static_cast<double>(e.weight);
            }
        }
        // Diagonal entry L(u, u) = sum_{v != u} w(u, v)
        rows.push_back(static_cast<idx>(u));
        cols.push_back(static_cast<idx>(u));
        vals.push_back(deg);
    }
    return SparseMatrix::from_triplets(static_cast<idx>(g.n_vertices()),
                                       static_cast<idx>(g.n_vertices()), rows, cols, vals);
}

template <typename Weight = real, typename Index = idx>
[[nodiscard]] inline Matrix dense_laplacian(const BasicGraph<Weight, Index> &g) {
    Matrix L(static_cast<idx>(g.n_vertices()), static_cast<idx>(g.n_vertices()), 0.0);
    for (Index u = 0; u < g.n_vertices(); ++u) {
        double deg = 0.0;
        for (const auto &e : g.neighbors(u)) {
            if (e.to != u) {
                L(static_cast<idx>(u), static_cast<idx>(e.to)) -= static_cast<double>(e.weight);
                deg += static_cast<double>(e.weight);
            }
        }
        L(static_cast<idx>(u), static_cast<idx>(u)) += deg;
    }
    return L;
}

template <typename Weight = real, typename Index = idx>
[[nodiscard]] inline SparseMatrix markov_generator(const BasicGraph<Weight, Index> &g,
                                                   bool column_oriented = true) {
    std::vector<idx> rows;
    std::vector<idx> cols;
    std::vector<double> vals;

    for (Index u = 0; u < g.n_vertices(); ++u) {
        double deg = 0.0;
        for (const auto &e : g.neighbors(u)) {
            if (e.to != u) {
                if (column_oriented) {
                    rows.push_back(static_cast<idx>(e.to));
                    cols.push_back(static_cast<idx>(u));
                } else {
                    rows.push_back(static_cast<idx>(u));
                    cols.push_back(static_cast<idx>(e.to));
                }
                vals.push_back(static_cast<double>(e.weight));
                deg += static_cast<double>(e.weight);
            }
        }
        rows.push_back(static_cast<idx>(u));
        cols.push_back(static_cast<idx>(u));
        vals.push_back(-deg);
    }
    return SparseMatrix::from_triplets(static_cast<idx>(g.n_vertices()),
                                       static_cast<idx>(g.n_vertices()), rows, cols, vals);
}

template <typename Weight = real, typename Index = idx>
[[nodiscard]] inline Matrix dense_markov_generator(const BasicGraph<Weight, Index> &g,
                                                   bool column_oriented = true) {
    Matrix Q(static_cast<idx>(g.n_vertices()), static_cast<idx>(g.n_vertices()), 0.0);
    auto L = dense_laplacian(g);
    for (idx i = 0; i < static_cast<idx>(g.n_vertices()); ++i) {
        for (idx j = 0; j < static_cast<idx>(g.n_vertices()); ++j) {
            Q(i, j) = column_oriented ? -L(i, j) : -L(j, i);
        }
    }
    return Q;
}

/// @brief Laplacian of a multigraph, summing parallel edge multiplicities.
[[nodiscard]] inline SparseMatrix laplacian(const structures::Multigraph &g) {
    const idx n = g.n_vertices();
    std::vector<idx> rows, cols;
    std::vector<real> vals;

    for (idx u = 0; u < n; ++u) {
        real diag = 0.0;
        for (const auto &e : g.adjacency()[u]) {
            if (e.to != u) {
                rows.push_back(static_cast<idx>(u));
                cols.push_back(static_cast<idx>(e.to));
                vals.push_back(-static_cast<real>(e.weight));
                diag += static_cast<real>(e.weight);
            }
        }
        rows.push_back(static_cast<idx>(u));
        cols.push_back(static_cast<idx>(u));
        vals.push_back(diag);
    }
    return SparseMatrix::from_triplets(static_cast<idx>(n), static_cast<idx>(n), rows, cols, vals);
}

/// @brief Recover a multigraph from a Laplacian.
[[nodiscard]] inline structures::Multigraph to_multigraph(const SparseMatrix &L) {
    const idx n = L.n_rows();
    if (L.n_cols() != n) {
        throw std::invalid_argument("to_multigraph: matrix must be square");
    }
    Multigraph mg(n);
    for (idx i = 0; i < n; ++i) {
        const idx row_start = L.row_ptr()[i];
        const idx row_end = L.row_ptr()[i + 1];
        for (idx p = row_start; p < row_end; ++p) {
            const idx j = L.col_idx()[p];
            const real val = L.values()[p];
            if (j > i && val < 0.0) {
                mg.add_edge(i, j, -val, 1);
            }
        }
    }
    return mg;
}

} // namespace num::linear
