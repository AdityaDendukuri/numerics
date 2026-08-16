/// @file 09_mcmc_bayesian_sampling.cpp
/// @brief Online Running Statistics, Histograms, and Autocorrelation Times.
#include <numerics.hpp>
#include <iostream>
#include <vector>
#include <random>

int main() {
    using namespace num;

    // 1. Running Statistics (Welford's Algorithm)
    RunningStats stats;
    std::mt19937 rng(42);
    std::normal_distribution<double> norm_dist(0.0, 1.0);

    for (int i = 0; i < 10000; ++i) {
        stats.update(norm_dist(rng));
    }
    std::cout << "Online Welford Mean = " << stats.mean << " (Target: 0.0), Variance = " << stats.variance() << " (Target: 1.0)\n";

    // 2. Fixed-Bin Histogram
    Histogram hist(20, -3.0, 3.0);
    for (int i = 0; i < 10000; ++i) {
        hist.fill(norm_dist(rng));
    }
    std::cout << "Histogram Total Counts = " << hist.total() << ", Bin Width = " << hist.bin_width() << "\n";

    // 3. Autocorrelation Time (Madras & Sokal)
    std::vector<real> time_series(1000);
    for (size_t i = 0; i < 1000; ++i) {
        time_series[i] = norm_dist(rng);
    }
    real tau_int = autocorr_time(time_series.data(), time_series.size());
    std::cout << "Autocorrelation Time tau_int = " << tau_int << " (uncorrelated limit: 0.5)\n";

    return 0;
}
