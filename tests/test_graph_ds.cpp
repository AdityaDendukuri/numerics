#include <gtest/gtest.h>
#include <numerics.hpp>
#include <random>

using namespace num;

// disjoint_set Tests (64-bit and 32-bit)

TEST(disjoint_set, InitializationAndSingletonProperties) {
    disjoint_set ds(10);
    EXPECT_EQ(ds.size(), 10);
    EXPECT_EQ(ds.count(), 10);
    for (idx i = 0; i < 10; ++i) {
        EXPECT_EQ(ds.find(i), i);
        EXPECT_EQ(ds.component_size(i), 1);
    }
}

TEST(disjoint_set, UnionFindAndPathCompression) {
    disjoint_set ds(6);
    EXPECT_TRUE(ds.unite(0, 1));
    EXPECT_TRUE(ds.unite(1, 2));
    EXPECT_FALSE(ds.unite(0, 2)); // already united

    EXPECT_TRUE(ds.connected(0, 2));
    EXPECT_EQ(ds.component_size(0), 3);
    EXPECT_EQ(ds.count(), 4);

    EXPECT_TRUE(ds.unite(3, 4));
    EXPECT_TRUE(ds.unite(4, 5));
    EXPECT_EQ(ds.count(), 2);

    EXPECT_FALSE(ds.connected(0, 5));
    EXPECT_TRUE(ds.unite(2, 3));
    EXPECT_TRUE(ds.connected(0, 5));
    EXPECT_EQ(ds.count(), 1);
    EXPECT_EQ(ds.component_size(4), 6);

    auto comps = ds.components();
    ASSERT_EQ(comps.size(), 1);
    EXPECT_EQ(comps[0].size(), 6);
}

TEST(disjoint_set, Templated32BitDisjointSet) {
    disjoint_set_32 ds(100);
    EXPECT_EQ(ds.size(), 100u);
    EXPECT_TRUE(ds.unite(10u, 20u));
    EXPECT_TRUE(ds.connected(10u, 20u));
    EXPECT_EQ(ds.component_size(10u), 2u);
}

TEST(disjoint_set, DiagnosticsBoundsCheck) {
    disjoint_set ds(5);
    EXPECT_THROW(ds.find(10), std::invalid_argument);
    EXPECT_THROW(ds.unite(0, 5), std::invalid_argument);
}

// indexed_priority_queue Tests

TEST(indexed_priority_queue, MinHeapOrderingAndKeyUpdates) {
    min_indexed_pq<double> pq(10);
    EXPECT_TRUE(pq.empty());

    pq.push(0, 5.0);
    pq.push(1, 2.5);
    pq.push(2, 8.0);
    pq.push(3, 1.0);

    EXPECT_EQ(pq.size(), 4);
    EXPECT_TRUE(pq.contains(2));
    EXPECT_FALSE(pq.contains(4));
    EXPECT_EQ(pq.top_index(), 3);
    EXPECT_DOUBLE_EQ(pq.top_key(), 1.0);

    // Decrease key of index 2 from 8.0 -> 0.5 (should become top)
    pq.improve_key(2, 0.5);
    EXPECT_EQ(pq.top_index(), 2);
    EXPECT_DOUBLE_EQ(pq.top_key(), 0.5);

    // Pop minimums in ascending order
    EXPECT_EQ(pq.top_index(), 2);
    pq.pop();
    EXPECT_EQ(pq.top_index(), 3);
    pq.pop();
    EXPECT_EQ(pq.top_index(), 1);
    pq.pop();
    EXPECT_EQ(pq.top_index(), 0);
    pq.pop();
    EXPECT_TRUE(pq.empty());
}

TEST(indexed_priority_queue, TemplatedFloatAndUint32) {
    min_indexed_pq<float, uint32_t> pq(10);
    pq.push(0u, 4.5f);
    pq.push(1u, 1.2f);
    EXPECT_EQ(pq.top_index(), 1u);
    EXPECT_FLOAT_EQ(pq.top_key(), 1.2f);
}

TEST(indexed_priority_queue, EraseAndClear) {
    min_indexed_pq<double> pq(5);
    pq.push(0, 10.0);
    pq.push(1, 20.0);
    pq.push(2, 5.0);

    pq.erase(2);
    EXPECT_FALSE(pq.contains(2));
    EXPECT_EQ(pq.top_index(), 0);

    pq.clear();
    EXPECT_TRUE(pq.empty());
}

TEST(indexed_priority_queue, DiagnosticsErrors) {
    min_indexed_pq<double> pq(5);
    pq.push(0, 1.0);
    EXPECT_THROW(pq.push(0, 2.0), std::invalid_argument); // Duplicate key
    EXPECT_THROW(pq.push(10, 1.0), std::invalid_argument); // Out of bounds
    EXPECT_THROW(pq.erase(1), std::invalid_argument); // Key not found
}

// graph and Algorithms Tests

TEST(graph, AdjacencyAndDegrees) {
    graph G(4);
    G.add_edge(0, 1, 2.0);
    G.add_edge(1, 2, 3.0);
    G.add_edge(2, 3, 1.5);

    EXPECT_EQ(G.n_vertices(), 4);
    EXPECT_EQ(G.n_edges(), 3);
    EXPECT_TRUE(G.has_edge(0, 1));
    EXPECT_TRUE(G.has_edge(1, 0));
    EXPECT_DOUBLE_EQ(G.edge_weight(0, 1), 2.0);
    EXPECT_EQ(G.degree(1), 2);
    EXPECT_DOUBLE_EQ(G.weighted_degree(1), 5.0);
}

TEST(graph, TemplatedFloatGraph) {
    float_graph G(3);
    G.add_edge(0u, 1u, 1.5f);
    G.add_edge(1u, 2u, 2.5f);

    EXPECT_EQ(G.n_vertices(), 3u);
    EXPECT_FLOAT_EQ(G.edge_weight(0u, 1u), 1.5f);
    EXPECT_FLOAT_EQ(G.weighted_degree(1u), 4.0f);
    EXPECT_TRUE(structures::is_connected(G));
}

TEST(graph, LaplacianAndMarkovGeneratorConversion) {
    graph G(3);
    G.add_edge(0, 1, 1.0);
    G.add_edge(1, 2, 2.0);

    mat L = num::linear::dense_laplacian(G);
    // L = [[1, -1, 0], [-1, 3, -2], [0, -2, 2]]
    EXPECT_DOUBLE_EQ(L(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(L(0, 1), -1.0);
    EXPECT_DOUBLE_EQ(L(1, 1), 3.0);
    EXPECT_DOUBLE_EQ(L(1, 2), -2.0);
    EXPECT_DOUBLE_EQ(L(2, 2), 2.0);

    mat Q = num::linear::dense_markov_generator(G, true);
    // Column sums of Q must be exactly 0
    for (idx j = 0; j < 3; ++j) {
        double col_sum = 0.0;
        for (idx i = 0; i < 3; ++i) col_sum += Q(i, j);
        EXPECT_NEAR(col_sum, 0.0, 1e-15);
    }
}

TEST(graph, ConnectivityAndDijkstra) {
    graph G(5);
    G.add_edge(0, 1, 4.0);
    G.add_edge(0, 2, 2.0);
    G.add_edge(1, 2, 1.0);
    G.add_edge(1, 3, 5.0);
    G.add_edge(2, 3, 8.0);
    G.add_edge(2, 4, 10.0);
    G.add_edge(3, 4, 2.0);

    EXPECT_TRUE(structures::is_connected(G));

    auto dist = structures::dijkstra(G, 0);
    EXPECT_DOUBLE_EQ(dist[0], 0.0);
    EXPECT_DOUBLE_EQ(dist[2], 2.0);
    EXPECT_DOUBLE_EQ(dist[1], 3.0); // 0 -> 2 -> 1
    EXPECT_DOUBLE_EQ(dist[3], 8.0); // 0 -> 2 -> 1 -> 3
    EXPECT_DOUBLE_EQ(dist[4], 10.0); // 0 -> 2 -> 1 -> 3 -> 4
}

TEST(graph, KruskalMST) {
    graph G(4);
    G.add_edge(0, 1, 1.0);
    G.add_edge(1, 2, 2.0);
    G.add_edge(2, 3, 3.0);
    G.add_edge(0, 3, 4.0);
    G.add_edge(0, 2, 5.0);

    graph mst = structures::minimum_spanning_tree(G);
    EXPECT_EQ(mst.n_edges(), 3);
    EXPECT_TRUE(structures::is_connected(mst));
    EXPECT_TRUE(mst.has_edge(0, 1));
    EXPECT_TRUE(mst.has_edge(1, 2));
    EXPECT_TRUE(mst.has_edge(2, 3));
    EXPECT_FALSE(mst.has_edge(0, 3));
}

TEST(graph, GeneratorsSpanningTreeAndErdosRenyi) {
    std::mt19937_64 rng(123);
    const idx n = 30;

    graph tree = structures::random_spanning_tree(n, rng, 0.5, 2.0);
    EXPECT_EQ(tree.n_vertices(), n);
    EXPECT_EQ(tree.n_edges(), n - 1);
    EXPECT_TRUE(structures::is_connected(tree));

    graph g_er = structures::erdos_renyi(n, 0.1, rng, true);
    EXPECT_EQ(g_er.n_vertices(), n);
    EXPECT_GE(g_er.n_edges(), n - 1);
    EXPECT_TRUE(structures::is_connected(g_er));
}

TEST(graph, CanonicalFamilies) {
    graph path = structures::path_graph(5);
    EXPECT_EQ(path.n_edges(), 4);
    EXPECT_TRUE(structures::is_connected(path));

    graph cycle = structures::cycle_graph(5);
    EXPECT_EQ(cycle.n_edges(), 5);

    graph grid = structures::grid_2d(3, 4);
    EXPECT_EQ(grid.n_vertices(), 12);
    EXPECT_EQ(grid.n_edges(), 2 * 3 * 4 - 3 - 4); // 24 - 7 = 17
    EXPECT_TRUE(structures::is_connected(grid));
}

// degree_queue Tests (ds module)

TEST(degree_queue, InsertPopMinAndRekey) {
    degree_queue dq(5);
    dq.insert(0, 3);
    dq.insert(1, 1);
    dq.insert(2, 4);
    dq.insert(3, 2);
    dq.insert(4, 5);

    EXPECT_EQ(dq.size(), 5u);
    EXPECT_EQ(dq.min_degree(), 1u);
    EXPECT_EQ(dq.pop_min(), 1u);

    // Rekey vertex 4 to degree 0
    dq.rekey(4, 0);
    EXPECT_EQ(dq.min_degree(), 0u);
    EXPECT_EQ(dq.pop_min(), 4u);

    EXPECT_EQ(dq.pop_min(), 3u); // degree 2
    EXPECT_EQ(dq.pop_min(), 0u); // degree 3
    EXPECT_EQ(dq.pop_min(), 2u); // degree 4
    EXPECT_TRUE(dq.empty());
}

TEST(degree_queue, TemplatedDegreeQueue32) {
    degree_queue_32 dq(10);
    for (uint32_t i = 0; i < 10; ++i) {
        dq.insert(i, 10 - i);
    }
    EXPECT_EQ(dq.pop_min(), 9u);
    EXPECT_EQ(dq.pop_min(), 8u);
}

// multigraph Tests (graph module)

TEST(multigraph, ParallelEdgesAndLaplacian) {
    multigraph mg(3);
    mg.add_edge(0, 1, 1.0, 2); // 2 parallel edges of weight 1.0
    mg.add_edge(1, 2, 3.0, 1);

    EXPECT_EQ(mg.n_vertices(), 3u);
    EXPECT_EQ(mg[0].size(), 1u);
    EXPECT_EQ(mg[0][0].to, 1u);
    EXPECT_EQ(mg[0][0].count, 2u);

    spmat L = num::linear::laplacian(mg);
    EXPECT_EQ(L.n_rows(), 3u);
    EXPECT_EQ(L.n_cols(), 3u);

    // Diagonal of L: vertex 0 has weight 1.0, vertex 1 has weight 1.0 + 3.0 = 4.0
    vec d = diagonal(L);
    EXPECT_DOUBLE_EQ(d[0], 1.0);
    EXPECT_DOUBLE_EQ(d[1], 4.0);
    EXPECT_DOUBLE_EQ(d[2], 3.0);
}
