/// @file stats/stats.cpp
#include "stats/stats.hpp"

namespace num {

real autocorr_time(const real* data, idx n, real c) {
    if (n < 4)
        return 0.5;

    // Compute mean
    real mean = 0.0;
    for (idx i = 0; i < n; ++i)
        mean += data[i];
    mean /= static_cast<real>(n);

    // C(0)  -- variance
    real c0 = 0.0;
    for (idx i = 0; i < n; ++i) {
        real d = data[i] - mean;
        c0 += d * d;
    }
    c0 /= static_cast<real>(n);
    if (c0 < 1e-15)
        return 0.5;

    // Accumulate C(t)/C(0) with automatic windowing
    real tau = 0.5;
    for (idx t = 1; t < n / 2; ++t) {
        real ct = 0.0;
        for (idx i = 0; i < n - t; ++i)
            ct += (data[i] - mean) * (data[i + t] - mean);
        ct /= static_cast<real>(n - t);
        tau += ct / c0;
        // Madras-Sokal window: stop when W > c * tau
        if (static_cast<real>(t) >= c * tau)
            break;
    }
    return (tau < 0.5) ? 0.5 : tau;
}

} // namespace num
