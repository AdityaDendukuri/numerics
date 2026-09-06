/// @file examples/12_hessenberg_resolvent_benchmark.cpp
/// @brief Benchmark comparing naive O(k*n^3) full LU batch resolvent against O(n^3 + k*n^2) Hessenberg resolvent.
#include <chrono>
#include <cmath>
#include <complex>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>
#include <numerics.hpp>

int main() {
    std::cout << "========================================================================\n";
    std::cout << "  NUMERICS RESOLVENT BENCHMARK: Naive O(k*N^3) LU vs O(N^3 + k*N^2) Hessenberg\n";
    std::cout << "========================================================================\n\n";

    const std::vector<num::idx> sizes = {50, 100, 200, 400};
    const std::size_t num_shifts = 100;

    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    // Generate 100 sample shifts along the imaginary axis (typical in transfer function & AAA sampling)
    std::vector<num::cplx> shifts(num_shifts);
    for (std::size_t k = 0; k < num_shifts; ++k) {
        shifts[k] = num::cplx(0.1, static_cast<double>(k) * 0.1);
    }

    std::cout << std::left << std::setw(8) << "N"
              << std::setw(10) << "Shifts"
              << std::setw(18) << "Naive LU (ms)"
              << std::setw(20) << "Hessenberg (ms)"
              << std::setw(12) << "Speedup"
              << std::setw(15) << "Max Diff" << "\n";
    std::cout << std::string(80, '-') << "\n";

    for (num::idx n : sizes) {
        num::mat A(n, n, 0.0);
        for (num::idx i = 0; i < n; ++i) {
            for (num::idx j = 0; j < n; ++j) {
                A(i, j) = dist(rng);
            }
            A(i, i) += 5.0; // Ensure well-conditioned
        }

        num::vec b(n);
        for (num::idx i = 0; i < n; ++i) {
            b[i] = dist(rng);
        }

        // 1. Naive O(k * n^3) approach: factorize sI - A fresh on every shift
        auto t0 = std::chrono::high_resolution_clock::now();
        std::vector<std::vector<num::cplx>> naive_sol(num_shifts);
        for (std::size_t k = 0; k < num_shifts; ++k) {
            num::dense_resolvent_solver naive_solver(A);
            naive_solver.factorize(shifts[k]);
            std::vector<num::cplx> b_cplx(n);
            for (num::idx i = 0; i < n; ++i) b_cplx[i] = b[i];
            naive_sol[k] = naive_solver.solve(b_cplx);
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double naive_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        // 2. Optimized O(n^3 + k * n^2) Hessenberg approach
        auto t2 = std::chrono::high_resolution_clock::now();
        num::hessenberg_resolvent_solver hess_solver(A);
        auto hess_sol = hess_solver.solve_batch(shifts, b);
        auto t3 = std::chrono::high_resolution_clock::now();
        double hess_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();

        // Check error agreement
        double max_diff = 0.0;
        for (std::size_t k = 0; k < num_shifts; ++k) {
            for (num::idx i = 0; i < n; ++i) {
                double diff = std::abs(naive_sol[k][i] - hess_sol[k][i]);
                if (diff > max_diff) max_diff = diff;
            }
        }

        double speedup = naive_ms / hess_ms;

        std::cout << std::left << std::setw(8) << n
                  << std::setw(10) << num_shifts
                  << std::setw(18) << std::fixed << std::setprecision(2) << naive_ms
                  << std::setw(20) << std::fixed << std::setprecision(2) << hess_ms
                  << std::setw(12) << std::fixed << std::setprecision(1) << (std::to_string(speedup) + "x")
                  << std::setw(15) << std::scientific << std::setprecision(2) << max_diff << "\n";
    }

    std::cout << "\n========================================================================\n";
    return 0;
}
