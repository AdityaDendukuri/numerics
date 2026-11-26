/// @file stats/stats.hpp
/// @brief Online statistics for Monte Carlo observables.
#pragma once
#include "stats/probability.hpp"
#include "stats/selection.hpp"

#include "core/types.hpp"
#include "stats/concepts.hpp"
#include <algorithm>
#include <cmath>
#include <concepts>
#include <vector>

namespace num {

/// @brief Mean and variance of a stream of samples, by Welford's method.
template <typename Float = double, std::integral Index = num::idx>
struct basic_running_stats {
    Float mean = Float{0};
    Float M2 = Float{0};
    Index count = 0;

    /// @brief Add the sample \f$x\f$.
    void update(Float x) {
        ++count;
        Float delta = x - mean;
        mean += delta / static_cast<Float>(count);
        Float delta2 = x - mean;
        M2 += delta * delta2;
    }

    /// @brief Return the unbiased sample variance: \f$s^2\f$; 0 with fewer than two samples.
    [[nodiscard]] Float variance() const {
        return (count < 2) ? Float{0} : M2 / static_cast<Float>(count - 1);
    }

    /// @brief Return the sample standard deviation: \f$s\f$.
    [[nodiscard]] Float std_dev() const { return std::sqrt(variance()); }

    /// @brief Return the standard error of the mean for uncorrelated samples: \f$s / \sqrt{n}\f$.
    [[nodiscard]] Float stderr_mean() const {
        return (count < 2) ? Float{0} : std_dev() / std::sqrt(static_cast<Float>(count));
    }

    /// @brief Discard all samples.
    void reset() {
        mean = M2 = Float{0};
        count = 0;
    }
};

using running_stats = basic_running_stats<real, idx>;

/// @brief Histogram with equal bins over \f$[\ell, h)\f$.
template <typename Float = double, std::integral Index = num::idx>
struct basic_histogram {
    array<Float> counts;
    Float lo = Float{0};
    Float hi = Float{0};
    Index nbins = 0;

    /// @brief Construct a histogram of `nbins` equal bins over \f$[\ell, h)\f$.
    basic_histogram(Index nbins, Float lo, Float hi)
        : counts(static_cast<std::size_t>(nbins), Float{0}), lo(lo), hi(hi), nbins(nbins) {}

    /// @brief Return the bin of \f$x\f$, or `nbins` outside \f$[\ell, h)\f$.
    [[nodiscard]] Index bin(Float x) const {
        if (x < lo || x >= hi) {
            return nbins;
        }
        return static_cast<Index>((x - lo) / (hi - lo) * static_cast<Float>(nbins));
    }

    /// @brief Return the centre of bin \f$b\f$.
    [[nodiscard]] Float bin_centre(Index b) const {
        return lo + ((static_cast<Float>(b) + Float{0.5}) * (hi - lo) / static_cast<Float>(nbins));
    }

    [[nodiscard]] Float bin_width() const { return (hi - lo) / static_cast<Float>(nbins); }

    /// @brief Add \f$x\f$ with weight `weight`, if it lies in the range.
    void fill(Float x, Float weight = Float{1.0}) {
        Index b = bin(x);
        if (b < nbins) {
            counts[static_cast<std::size_t>(b)] += weight;
        }
    }

    void reset() { std::fill(counts.begin(), counts.end(), Float{0}); }

    /// @brief Return the total weight.
    [[nodiscard]] Float total() const {
        Float s = Float{0};
        for (Float c : counts) {
            s += c;
        }
        return s;
    }

    /// @brief Return the counts scaled to a probability density: \f$\int p = 1\f$.
    [[nodiscard]] array<Float> pdf() const {
        Float norm = total() * bin_width();
        array<Float> p(static_cast<std::size_t>(nbins));
        if (norm > Float{0}) {
            for (Index b = 0; b < nbins; ++b) {
                p[static_cast<std::size_t>(b)] = counts[static_cast<std::size_t>(b)] / norm;
            }
        }
        return p;
    }
};

using histogram = basic_histogram<real, idx>;

// Autocorrelation time
/// @brief Estimate the integrated autocorrelation time:
/// \f$\tau = \tfrac{1}{2} + \sum_{t \ge 1} \rho(t)\f$.
template <typename Float = double, std::integral Index = num::idx>
inline Float autocorr_time(const Float *data, Index n, Float c = Float{6.0}) {
    if (n < 4) {
        return Float{0.5};
    }

    Float mean = Float{0};
    for (Index i = 0; i < n; ++i) {
        mean += data[i];
    }
    mean /= static_cast<Float>(n);

    Float c0 = Float{0};
    for (Index i = 0; i < n; ++i) {
        Float d = data[i] - mean;
        c0 += d * d;
    }
    c0 /= static_cast<Float>(n);
    if (c0 < Float{1e-15}) {
        return Float{0.5};
    }

    Float tau = Float{0.5};
    for (Index t = 1; t < n / 2; ++t) {
        Float ct = Float{0};
        for (Index i = 0; i < n - t; ++i) {
            ct += (data[i] - mean) * (data[i + t] - mean);
        }
        ct /= static_cast<Float>(n - t);
        tau += ct / c0;
        if (static_cast<Float>(t) >= c * tau) {
            break;
        }
    }
    return (tau < Float{0.5}) ? Float{0.5} : tau;
}

static_assert(moment_accumulator<running_stats>);
} // namespace num
