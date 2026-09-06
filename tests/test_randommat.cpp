#include <gtest/gtest.h>
#include <numerics.hpp>
#include <random>

using namespace num;

TEST(RandomMat, ApproxCholBasicGraphFactorizeSolve) {
    // 4-node cycle graph
    graph G(4);
    G.add_edge(0, 1, 2.0);
    G.add_edge(1, 2, 2.0);
    G.add_edge(2, 3, 2.0);
    G.add_edge(3, 0, 2.0);

    auto ac_G = to_approxchol_graph(G);
    ASSERT_EQ(ac_G.size(), 4u);

    auto factor_ac1 = randommat::ac1(ac_G, 42);
    EXPECT_EQ(factor_ac1.order.size(), 4u);

    auto factor_exact = randommat::exact(ac_G);
    EXPECT_EQ(factor_exact.order.size(), 4u);

    // Solve against zero-sum RHS (orthogonal to nullspace)
    vec b = {1.0, -1.0, 1.0, -1.0};
    vec x(4, 0.0);
    randommat::solve(factor_exact, b, x);

    // Check L * x = b for the exact factor
    spmat L = num::linear::laplacian(G);
    vec Lx(4, 0.0);
    sparse_matvec(L, x, Lx);

    for (idx i = 0; i < 4; ++i) {
        EXPECT_NEAR(Lx[i], b[i], 1e-12);
    }
}

TEST(RandomMat, ApproxCholPreconditionerWithPCG) {
    const idx n = 20;
    std::mt19937_64 rng(12345);

    // Connected path/cycle graph
    graph G(n);
    for (idx i = 0; i < n - 1; ++i) {
        G.add_edge(i, i + 1, 1.5);
    }
    G.add_edge(0, n - 1, 0.5);

    mat L = num::linear::dense_laplacian(G);
    operators::dense_op op(L);
    const space::zero_sum zero_sum_space{};
    const auto laplacian_on_zero_sum =
        num::assume<law::spd_on<space::zero_sum>>(op);

    // 1. Exact factor PCG -> 1 step solve
    auto ac_G = to_approxchol_graph(G);
    auto exact_factor = randommat::exact(ac_G);
    approx_chol_preconditioner exact_prec(exact_factor);
    const auto projected_exact_prec = operators::projected(exact_prec, zero_sum_space);
    const auto exact_prec_on_zero_sum =
        num::assume<law::spd_on<space::zero_sum>>(projected_exact_prec);

    vec b(n, 0.0);
    b[0] = 1.0;
    b[n - 1] = -1.0;

    vec x_exact(n, 0.0);
    auto res_exact = pcg(laplacian_on_zero_sum, exact_prec_on_zero_sum, b, x_exact, zero_sum_space,
                         {.tolerance = 1e-12, .max_iterations = 10});
    EXPECT_TRUE(res_exact.converged);
    EXPECT_LE(res_exact.iterations, 2u);
    EXPECT_LT(res_exact.residual, 1e-11);

    // 2. Sampled ApproxChol preconditioner
    auto sampled_prec = approxchol_preconditioner(G, /*samples=*/2, /*seed=*/42);
    const auto projected_sampled_prec = operators::projected(sampled_prec, zero_sum_space);
    const auto sampled_prec_on_zero_sum =
        num::assume<law::spd_on<space::zero_sum>>(projected_sampled_prec);
    vec x_sampled(n, 0.0);
    auto res_sampled = pcg(laplacian_on_zero_sum, sampled_prec_on_zero_sum, b, x_sampled,
                           zero_sum_space, {.tolerance = 1e-8, .max_iterations = 50});
    EXPECT_TRUE(res_sampled.converged);
    EXPECT_LT(res_sampled.residual, 1e-7);
}

TEST(RandomMat, SparseMatrixConversion) {
    graph G(5);
    G.add_edge(0, 1, 1.5);
    G.add_edge(1, 2, 2.5);
    G.add_edge(2, 3, 3.5);
    G.add_edge(3, 4, 4.5);
    G.add_edge(4, 0, 0.5);

    spmat L = num::linear::laplacian(G);
    auto ac_prec = approxchol_preconditioner(L, 1, 42);
    EXPECT_EQ(ac_prec.rows(), 5u);
    EXPECT_EQ(ac_prec.cols(), 5u);
}
