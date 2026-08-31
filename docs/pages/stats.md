# Statistics {#page_stats}

Streaming accumulators for moments (Welford's algorithm) and empirical probability density histograms.

---

## 1. Running Statistics (num::RunningStats)

Single-pass \f$\mathcal{O}(1)\f$ memory accumulation of mean, variance, and standard error using Welford's recurrence:
\f[
\bar{x}_k = \bar{x}_{k-1} + \frac{x_k - \bar{x}_{k-1}}{k}, \qquad M_{2,k} = M_{2,k-1} + (x_k - \bar{x}_{k-1})(x_k - \bar{x}_k)
\f]

```cpp
#include <numerics.hpp>

num::RunningStats stats;

for (double x : samples) {
    stats.update(x);
}

double mean        = stats.mean;
double variance    = stats.variance();    // s^2 = M2 / (n - 1)
double std_dev     = stats.std_dev();     // s = sqrt(variance)
double stderr_mean = stats.stderr_mean(); // s / sqrt(n)
```

---

## 2. Empirical Histogram (num::Histogram)

Binned empirical probability density function (PDF).

```cpp
num::Histogram hist(/*n_bins=*/100, /*min=*/-1.0, /*max=*/1.0);

for (double x : samples) {
    hist.fill(x);
}

std::vector<double> pdf = hist.pdf(); // Normalized so sum(p_i * dx) == 1.0
```

---

## 3. Concepts

```cpp
static_assert(num::StreamingAccumulator<num::RunningStats>);
static_assert(num::MomentAccumulator<num::RunningStats>);
```

