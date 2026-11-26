/// @file 21_graphs.cpp
/// @brief Build and search graphs, find spanning trees, and solve a graph Laplacian system.
///
/// `graph<W, I>` stores weighted adjacency lists. `structures::dijkstra`, `bfs`,
/// `connected_components` and `minimum_spanning_tree` run on it directly, and
/// `linear::laplacian` turns it into a sparse matrix. A graph Laplacian is singular, so the
/// solve is posed on the zero-sum subspace, where it is positive definite, and preconditioned by
/// the randomized approximate Cholesky factorization of Gao, Kyng and Spielman.
#include <io/json.hpp>
#include <numerics.hpp>

using namespace num;

int main() {
    // A 40 x 40 grid with unit weights.
    const idx side = 40, n = side * side;
    const graph<real, idx> G = structures::grid_2d<real, idx>(side, side);
    const bool connected = structures::is_connected(G);

    const array<real> distance = structures::dijkstra(G, idx{0});


    const graph<real, idx> tree = structures::minimum_spanning_tree(G);
    real weight = 0.0;
    for (idx u = 0; u < n; ++u) {
        for (const auto &e : tree.neighbors(u)) {
            weight += u < e.to ? e.weight : 0.0;
        }
    }


    disjoint_set sets(6);
    sets.unite(0, 1);
    sets.unite(2, 3);
    sets.unite(1, 3);


    // L x = b on the zero-sum subspace: unit current in at one corner and out at the other.
    const mat<real> L = linear::dense_laplacian(G);
    const space::zero_sum zero_sum{};
    const auto laplacian = assume<law::spd_on<space::zero_sum>>(operators::dense_op(L));
    vec<real> b(n, 0.0), x(n, 0.0);
    b[0] = 1.0;
    b[n - 1] = -1.0;

    // projected() holds a reference, so the factor it wraps is named rather than a temporary.
    const auto approxchol = approxchol_preconditioner(G, gao_kyng_spielman_2023::ac2, 42);
    const auto preconditioner =
        assume<law::spd_on<space::zero_sum>>(operators::projected(approxchol, zero_sum));
    const auto result = pcg(laplacian, preconditioner, b, x, zero_sum,
                            {.tolerance = {.rtol = 1e-8}, .max_iterations = 200});
    io::write_json({
        {"vertices", n}, {"connected", connected},
        {"corner_distance", distance[n - 1]}, {"spanning_tree_weight", weight},
        {"disjoint_set", {{"zero_to_three", sets.connected(0, 3)},
                          {"zero_to_five", sets.connected(0, 5)}}},
        {"laplacian_pcg", {{"converged", result.converged},
                           {"iterations", result.iterations},
                           {"effective_resistance", x[0] - x[n - 1]}}}
    }, "result.json");
}
