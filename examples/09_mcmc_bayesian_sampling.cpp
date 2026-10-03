/// @file 09_mcmc_bayesian_sampling.cpp
/// @brief Running statistics and histograms over a stream of samples.
///
/// `running_stats` accumulates the mean and variance in one pass with Welford's update, which
/// stays accurate where the textbook sum-of-squares formula cancels. `histogram` bins samples
/// over a fixed range and returns a normalized density.
#include <cmath>
#include <cstdio>
#include <numbers>
#include <numerics.hpp>
#include <random>
#include <string_view>
#include <vector>

using namespace num;

int main(int argc, char **argv) {
    const bool plot = argc > 1 && std::string_view(argv[1]) == "--plot";

    std::mt19937 rng(42);
    std::normal_distribution<double> normal(0.0, 1.0);
    running_stats stats;
    histogram hist(24, -3.0, 3.0);
    for (int i = 0; i < 100000; ++i) {
        const double x = normal(rng);
        stats.update(x);
        hist.fill(x);
    }
    std::printf("100000 samples of N(0, 1): mean %+.4f, variance %.4f, standard error %.4f\n",
                stats.mean, stats.variance(), stats.stderr_mean());

    // Welford keeps its accuracy when the variance is tiny next to the mean.
    running_stats shifted;
    for (int i = 0; i < 1000; ++i) {
        shifted.update(1e9 + (i % 2 ? 1e-3 : -1e-3));
    }
    std::printf("samples 1e9 +- 1e-3: variance %.3e (exact %.3e)\n", shifted.variance(),
                1e-6 * 1000.0 / 999.0);

    const auto density = hist.pdf();
    std::printf("\n   x      density   N(0, 1)\n");
    for (idx b = 0; b < hist.nbins; b += 4) {
        const double x = hist.bin_centre(b);
        std::printf("%6.2f   %.4f    %.4f\n", x, density[b],
                    std::exp(-0.5 * x * x) / std::sqrt(2.0 * std::numbers::pi));
    }

    if (plot) {
        std::vector<double> x, p;
        for (idx b = 0; b < hist.nbins; ++b) {
            x.push_back(hist.bin_centre(b));
            p.push_back(density[b]);
        }
        plt::plot(x, p, "histogram density", "linespoints");
        plt::title("09 Sample density");
        plt::show();
    }
}
