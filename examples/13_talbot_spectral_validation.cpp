/// @file 13_talbot_spectral_validation.cpp
/// @brief Talbot contour inversion checked against Krylov expv.
///
/// Talbot quadrature inverts a Laplace transform by summing resolvent solves
/// \f$(sI - Q)^{-1}p_0\f$ at shifts on a contour. Here Q is a random Markov generator with 100
/// states, the shifts all come from one Hessenberg reduction, and the result is compared with
/// \f$e^{tQ}p_0\f$ from `expv`. The error falls quickly with the number of nodes and reaches
/// rounding level at about 24.
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>
#include <numerics.hpp>
#include <string_view>

int main(int argc, char **argv) {
    const bool plot = argc > 1 && std::string_view(argv[1]) == "--plot";
    using namespace num;

    const idx N = 100;
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> weight_dist(0.5, 2.5);
    std::uniform_real_distribution<double> prob_dist(0.0, 1.0);

    // A connected Markov generator Q. Its columns sum to zero.
    mat<real> Q(N, N, 0.0);
    for (idx i = 0; i < N; ++i) {
        for (idx j = i + 1; j < N; ++j) {
            if (prob_dist(rng) < 0.10 || j == i + 1) { // keep the chain connected
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

    vec<real> p0(N, 0.0);
    p0[0] = 1.0;

    // One O(N^3) reduction serves every shift.
    hessenberg_resolvent hess_solver(Q);

    const double t_eval = 1.0;
    std::cout << "p(t) = e^{tQ} p0 at t = " << t_eval << ", N = " << N << " states\n";

    // The reference solution, from expv.
    operators::dense_op Q_op(Q);
    auto t_krylov_start = std::chrono::high_resolution_clock::now();
    vec<real> p_exact = expv(t_eval, Q_op, p0, 50, 1e-15);
    auto t_krylov_end = std::chrono::high_resolution_clock::now();
    double krylov_ms = std::chrono::duration<double, std::milli>(t_krylov_end - t_krylov_start).count();

    std::cout << "expv reference in " << std::fixed << std::setprecision(2) << krylov_ms << " ms\n\n";

    // Talbot with M nodes.
    const std::vector<idx> modes = {4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32};
    std::vector<double> modes_dbl, err_inf_list, err_l1_list;

    std::cout << std::left << std::setw(12) << "Nodes M"
              << std::setw(18) << "Time (ms)"
              << std::setw(18) << "L_inf error"
              << std::setw(18) << "L_1 error" << "\n";
    std::cout << std::string(66, '-') << "\n";

    for (idx M : modes) {
        auto t0 = std::chrono::high_resolution_clock::now();
        std::vector<cplx> density(N, cplx(0.0, 0.0));

        for (const auto &[shift, weight] : real_talbot_contour(t_eval, M)) {
            auto sol = solve(num::shift(hess_solver, shift), p0);
            for (idx i = 0; i < N; ++i) {
                density[i] += weight * sol[i];
            }
        }

        vec<real> p_talbot(N, 0.0);
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

    if (plot) {
        plt::plot(modes_dbl, err_inf_list, "L_inf error", "linespoints");
        plt::plot(modes_dbl, err_l1_list, "L_1 error", "linespoints");
        plt::title("13 Talbot error against expv, t = 1");
        plt::xlabel("Talbot nodes M");
        plt::ylabel("error");
        plt::semilogy();
        plt::legend();
        plt::show();
    }

    return 0;
}
