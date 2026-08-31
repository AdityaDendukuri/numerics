/// @file 09_mcmc_bayesian_sampling.cpp
/// @brief Online Running Statistics, Histograms, and Autocorrelation Times.
#include <iostream>
#include <numerics.hpp>
#include <random>
#include <vector>

int main() {
    using namespace num;

    RunningStats stats;
    std::mt19937 rng(42);
    std::normal_distribution<double> norm_dist(0.0, 1.0);

    Histogram hist(20, -3.0, 3.0);
    for (int i = 0; i < 10000; ++i) {
        double val = norm_dist(rng);
        stats.update(val);
        hist.fill(val);
    }
    std::cout << "Online Welford Mean = " << stats.mean
              << " (Target: 0.0), Variance = " << stats.variance() << " (Target: 1.0)\n";

    // Extract PDF and plot histogram
    auto pdf = hist.pdf();
    std::vector<double> bin_centers, pdf_vals;
    for (idx b = 0; b < hist.nbins; ++b) {
        bin_centers.push_back(hist.bin_centre(b));
        pdf_vals.push_back(pdf[b]);
    }

    plt::plot(bin_centers, pdf_vals, "Sample PDF", "linespoints");
    plt::title("09 Statistics: Sample Probability Density Function (PDF)");
    plt::xlabel("Sample Value x");
    plt::ylabel("Probability Density P(x)");
    plt::show_dumb(140, 35);

    return 0;
}
