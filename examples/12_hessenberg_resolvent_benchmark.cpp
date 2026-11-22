/// @file examples/12_hessenberg_resolvent_benchmark.cpp
/// @brief Benchmark comparing naive O(k*n^3) full LU batch resolvent against O(n^3 + k*n^2)
/// Hessenberg resolvent.
#include <chrono>
#include <cmath>
#include <complex>
#include <io/json.hpp>
#include <numerics.hpp>
#include <random>
#include <vector>

int main() {
    const std::vector<num::idx> sizes = {50, 100, 200, 400};
    const std::size_t num_shifts = 100;

    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    // Generate 100 sample shifts along the imaginary axis (typical in transfer function & AAA
    // sampling)
    std::vector<num::cplx> shifts(num_shifts);
    for (std::size_t k = 0; k < num_shifts; ++k) {
        shifts[k] = num::cplx(0.1, static_cast<double>(k) * 0.1);
    }

    num::io::json results = num::io::json::array();

    for (num::idx n : sizes) {
        num::mat<num::real> A(n, n, 0.0);
        for (num::idx i = 0; i < n; ++i) {
            for (num::idx j = 0; j < n; ++j) {
                A(i, j) = dist(rng);
            }
            A(i, i) += 5.0; // Ensure well-conditioned
        }

        num::vec<num::real> b(n);
        for (num::idx i = 0; i < n; ++i) {
            b[i] = dist(rng);
        }

        // Recompute the reduction for each shift.
        auto t0 = std::chrono::high_resolution_clock::now();
        std::vector<num::vec<num::cplx>> naive_sol(num_shifts);
        for (std::size_t k = 0; k < num_shifts; ++k) {
            num::hessenberg_resolvent naive_solver(A);
            naive_sol[k] = num::solve(num::shift(naive_solver, shifts[k]), b);
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double naive_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        // Reuse one reduction for all shifts.
        auto t2 = std::chrono::high_resolution_clock::now();
        num::hessenberg_resolvent hess_solver(A);
        auto hess_sol = num::solve_batch(hess_solver, shifts, b);
        auto t3 = std::chrono::high_resolution_clock::now();
        double hess_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();

        // Check error agreement
        double max_diff = 0.0;
        for (std::size_t k = 0; k < num_shifts; ++k) {
            for (num::idx i = 0; i < n; ++i) {
                double diff = std::abs(naive_sol[k][i] - hess_sol[k][i]);
                if (diff > max_diff)
                    max_diff = diff;
            }
        }

        double speedup = naive_ms / hess_ms;

        results.push_back({{"size", n}, {"shifts", num_shifts}, {"naive_ms", naive_ms},
                           {"hessenberg_ms", hess_ms}, {"speedup", speedup},
                           {"max_difference", max_diff}});
    }

    num::io::write_json(results, "result.json");
    return 0;
}
