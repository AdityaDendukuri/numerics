/// @file stats/stats.hpp
/// @brief Online statistics for Monte Carlo observables
///
/// RunningStats: Welford online mean + variance, no data storage needed.
/// Histogram:    Fixed-bin histogram with reweighting for WHAM analysis.
/// autocorr_time: Integrated autocorrelation time from a stored time series.
#pragma once

#include "core/types.hpp"
#include <vector>
#include <cmath>

namespace num {

// RunningStats
/// Online mean and variance via Welford's algorithm.
/// One-pass, numerically stable, O(1) memory.
struct RunningStats {
    real  mean  = 0.0;
    real  M2    = 0.0;
    idx   count = 0;

    /// Incorporate one new sample.
    void update(real x) {
        ++count;
        real delta  = x - mean;
        mean       += delta / static_cast<real>(count);
        real delta2 = x - mean;
        M2         += delta * delta2;
    }

    /// Unbiased sample variance (n-1 denominator). Returns 0 for n < 2.
    real variance() const {
        return (count < 2) ? 0.0 : M2 / static_cast<real>(count - 1);
    }

    real std_dev()     const { return std::sqrt(variance()); }

    /// Standard error of the mean (uncorrelated samples).
    real stderr_mean() const {
        return (count < 2) ? 0.0 : std_dev() / std::sqrt(static_cast<real>(count));
    }

    void reset() { mean = M2 = 0.0; count = 0; }
};

// Histogram
/// Fixed-bin histogram over [lo, hi).
/// Useful for umbrella sampling  -- each window collects a Histogram of the
/// reaction coordinate (e.g. nucleus size), then WHAM stitches them together.
struct Histogram {
    std::vector<real> counts;
    real lo, hi;
    idx  nbins;

    /// @param nbins  Number of bins
    /// @param lo,hi  Range of the histogram [lo, hi)
    Histogram(idx nbins, real lo, real hi)
        : counts(nbins, 0.0), lo(lo), hi(hi), nbins(nbins) {}

    /// Map value to bin index. Returns nbins (out of range sentinel) if outside.
    idx bin(real x) const {
        if (x < lo || x >= hi) return nbins;
        return static_cast<idx>((x - lo) / (hi - lo) * static_cast<real>(nbins));
    }

    real bin_centre(idx b) const {
        return lo + (static_cast<real>(b) + 0.5) * (hi - lo) / static_cast<real>(nbins);
    }

    real bin_width() const { return (hi - lo) / static_cast<real>(nbins); }

    void fill(real x, real weight = 1.0) {
        idx b = bin(x);
        if (b < nbins) counts[b] += weight;
    }

    void reset() { std::fill(counts.begin(), counts.end(), 0.0); }

    real total() const {
        real s = 0.0;
        for (real c : counts) s += c;
        return s;
    }

    /// Normalise so that the histogram integrates to 1 (probability density).
    std::vector<real> pdf() const {
        real norm = total() * bin_width();
        std::vector<real> p(nbins);
        if (norm > 0.0)
            for (idx b = 0; b < nbins; ++b) p[b] = counts[b] / norm;
        return p;
    }
};

// Autocorrelation time
/// Integrated autocorrelation time tau_int from a stored time series.
///
/// Uses the automatic windowing criterion (Madras & Sokal 1988):
///   accumulate C(t)/C(0) until W > c*tau_int  (c = 6 default).
/// Returns tau_int >= 0.5; returns 0.5 on failure.
///
/// @param data     Pointer to time series of length n
/// @param n        Length of the series
/// @param c        Window parameter (default 6)
real autocorr_time(const real* data, idx n, real c = 6.0);

} // namespace num
