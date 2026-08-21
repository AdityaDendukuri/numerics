/// @file stats/stats.hpp
/// @brief Online statistics for Monte Carlo observables.
#pragma once
#include "stats/probability.hpp"
#include "stats/selection.hpp"

#include "core/types.hpp"
#include <cmath>
#include <vector>

namespace num {

/// @brief Welford updates for mean and variance.
struct RunningStats {
  real mean = 0.0;
  real M2 = 0.0;
  idx count = 0;

  /// Incorporate one observation without storing the sample history.
  void update(real x) {
    ++count;
    real delta = x - mean;
    mean += delta / static_cast<real>(count);
    real delta2 = x - mean;
    M2 += delta * delta2;
  }

  /// Return the unbiased sample variance, or zero with fewer than two samples.
  [[nodiscard]] real variance() const {
    return (count < 2) ? 0.0 : M2 / static_cast<real>(count - 1);
  }

  /// Return the sample standard deviation.
  [[nodiscard]] real std_dev() const { return std::sqrt(variance()); }

  /// Return the uncorrelated standard error of the mean.
  [[nodiscard]] real stderr_mean() const {
    return (count < 2) ? 0.0 : std_dev() / std::sqrt(static_cast<real>(count));
  }

  /// Discard all accumulated observations.
  void reset() {
    mean = M2 = 0.0;
    count = 0;
  }
};

/// @brief Fixed-bin histogram over \f$[\ell,h)\f$.
struct Histogram {
  std::vector<real> counts;
  real lo = 0.0;
  real hi = 0.0;
  idx nbins = 0;

  /// Divide [lo,hi) into equally sized bins.
  Histogram(idx nbins, real lo, real hi)
      : counts(nbins, 0.0),
        lo(lo),
        hi(hi),
        nbins(nbins) {}

  /// Return the containing bin, or nbins when x lies outside the interval.
  [[nodiscard]] idx bin(real x) const {
    if (x < lo || x >= hi) {
      return nbins;
    }
    return static_cast<idx>((x - lo) / (hi - lo) * static_cast<real>(nbins));
  }

  /// Return the center coordinate of bin b.
  [[nodiscard]] real bin_centre(idx b) const {
    return lo + ((static_cast<real>(b) + 0.5) * (hi - lo) / static_cast<real>(nbins));
  }

  [[nodiscard]] real bin_width() const { return (hi - lo) / static_cast<real>(nbins); }

  /// Accumulate a weighted observation when it lies inside the interval.
  void fill(real x, real weight = 1.0) {
    idx b = bin(x);
    if (b < nbins) {
      counts[b] += weight;
    }
  }

  void reset() { std::fill(counts.begin(), counts.end(), 0.0); }

  /// Return the total accumulated bin weight.
  [[nodiscard]] real total() const {
    real s = 0.0;
    for (real c : counts) {
      s += c;
    }
    return s;
  }

  /// Normalise so that the histogram integrates to 1 (probability density).
  [[nodiscard]] std::vector<real> pdf() const {
    real norm = total() * bin_width();
    std::vector<real> p(nbins);
    if (norm > 0.0) {
      for (idx b = 0; b < nbins; ++b) {
        p[b] = counts[b] / norm;
      }
    }
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
