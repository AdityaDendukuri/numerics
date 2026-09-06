/// @file 13_talbot_spectral_validation.cpp
/// @brief High-precision validation of Talbot contour inversion against Krylov Arnoldi expv using num:: toolkit.
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>
#include <numerics.hpp>

int main() {
    using namespace num;

    std::cout << "========================================================================\n";
    std::cout << "  NUMERICS: TALBOT CONTOUR VS KRYLOV ARNOILDI EXPV VALIDATION\n";
    std::cout << "========================================================================\n\n";

    const idx N = 100;
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> weight_dist(0.5, 2.5);
    std::uniform_real_distribution<double> prob_dist(0.0, 1.0);

    // 1. Construct a connected Markov jump generator Q with row sums = 0
    mat Q(N, N, 0.0);
    for (idx i = 0; i < N; ++i) {
        for (idx j = i + 1; j < N; ++j) {
            if (prob_dist(rng) < 0.10 || j == i + 1) { // ensure connected chain
                double rate_ij = weight_dist(rng);
                double rate_ji = weight_dist(rng);
                Q(i, j) = rate_ij;
                Q(j, i) = rate_ji;
            }
        }
    }
    for (idx j = 0; j < N; ++j) {
        double col_rate = 0.0;
        for (idx i = 0; i < N; ++i) {
            if (i != j) col_rate += Q(i, j);
        }
        Q(j, j) = -col_rate;
    }

    vec p0(N, 0.0);
    p0[0] = 1.0; // Initial state at node 0

    // 2. Precompute Hessenberg decomposition of Q once in O(N^3)
    hessenberg_resolvent_solver hess_solver(Q);

    const double t_eval = 1.0;
    std::cout << "Evaluating Markov diffusion at t = " << t_eval << " (N = " << N << " states)...\n";

    // 3. High-precision Ground Truth via Krylov Arnoldi expv
    operators::dense_op Q_op(Q);
    auto t_krylov_start = std::chrono::high_resolution_clock::now();
    vec p_exact = expv(t_eval, Q_op, p0, 50, 1e-15);
    auto t_krylov_end = std::chrono::high_resolution_clock::now();
    double krylov_ms = std::chrono::duration<double, std::milli>(t_krylov_end - t_krylov_start).count();

    std::cout << "Krylov-Arnoldi ground truth computed in " << std::fixed << std::setprecision(2) << krylov_ms << " ms.\n\n";

    // 4. Parameter sweep over Talbot quadrature modes M
    const std::vector<idx> modes = {4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32};
    std::vector<double> modes_dbl, err_inf_list, err_l1_list;

    std::cout << std::left << std::setw(12) << "Nodes (M)"
              << std::setw(18) << "Walltime (ms)"
              << std::setw(18) << "L_inf Error"
              << std::setw(18) << "L_1 Error" << "\n";
    std::cout << std::string(66, '-') << "\n";

    for (idx M : modes) {
        auto t0 = std::chrono::high_resolution_clock::now();
        std::vector<cplx> density(N, cplx(0.0, 0.0));

        inverse_laplace_accumulate(t_eval, M, [&](cplx shift, cplx weight) {
            auto sol = hess_solver.solve(shift, p0);
            for (idx i = 0; i < N; ++i) {
                density[i] += weight * sol[i];
            }
        });

        vec p_talbot(N, 0.0);
        for (idx i = 0; i < N; ++i) p_talbot[i] = std::max(0.0, density[i].real());
        clip_and_normalize_nonnegative(p_talbot);

        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        double l_inf = 0.0, l_1 = 0.0;
        for (idx i = 0; i < N; ++i) {
            double diff = std::abs(p_talbot[i] - p_exact[i]);
            l_inf = std::max(l_inf, diff);
            l_1 += diff;
        }

        modes_dbl.push_back(static_cast<double>(M));
        err_inf_list.push_back(std::max(1e-16, l_inf));
        err_l1_list.push_back(std::max(1e-16, l_1));

        std::cout << std::left << std::setw(12) << M
                  << std::setw(18) << std::fixed << std::setprecision(3) << ms
                  << std::setw(18) << std::scientific << std::setprecision(3) << l_inf
                  << std::setw(18) << std::scientific << std::setprecision(3) << l_1 << "\n";
    }

    // 5. Multi-panel Visualization
    std::cout << "\nRendering ASCII convergence profile...\n";
    plt::plot(modes_dbl, err_inf_list, "L_inf Error", "linespoints");
    plt::plot(modes_dbl, err_l1_list, "L_1 Error", "linespoints");
    plt::title("13 Talbot Spectral Convergence vs Modes M (t = 1.0)");
    plt::xlabel("Talbot Quadrature Nodes M");
    plt::ylabel("Absolute Error vs Arnoldi expv");
    plt::semilogy();
    plt::legend();
    plt::show_dumb(120, 25);

    plt::savefig("13_talbot_spectral_validation.png");
    std::cout << "\nSaved high-resolution plot to 13_talbot_spectral_validation.png\n";
    std::cout << "========================================================================\n";

    return 0;
}
