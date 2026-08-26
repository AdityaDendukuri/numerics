# Statistics {#page_stats}

The `stats` module provides streaming accumulators for moments and empirical densities, plus selection helpers.

An accumulator absorbs samples one at a time at constant cost and constant storage. Anything that retains the samples in order to summarize them is not one.

---

## 1. Numerically Stable Running Statistics (Welford's Algorithm)

Computes sample mean \f$\bar{x}_n\f$ and sample variance \f$s_n^2\f$ in a single memory pass without catastrophic numerical cancellation:

\f[
\bar{x}_k = \bar{x}_{k-1} + \frac{x_k - \bar{x}_{k-1}}{k}, \qquad M_{2,k} = M_{2,k-1} + (x_k - \bar{x}_{k-1})(x_k - \bar{x}_k)
\f]

\f[
s_n^2 = \frac{M_{2,n}}{n - 1}, \qquad \text{SE}(\bar{x}) = \frac{s_n}{\sqrt{n}}
\f]

```cpp
num::RunningStats stats;

for (double x : samples) {
    stats.update(x); // Single-pass O(1) memory accumulation
}

double mean = stats.mean;
double variance = stats.variance();      // Unbiased sample variance s^2
double std_dev = stats.std_dev();        // Standard deviation s
double stderr_mean = stats.stderr_mean();// Standard error s / sqrt(n)
```

---

## 2. Histograms & Empirical Probability Densities

Discretizes sample spaces into uniform bins \f$[a + i \Delta x, a + (i+1)\Delta x)\f$ and normalizes to an empirical probability density function (PDF):

\f[
\int_{-\infty}^\infty p(x)\,dx \approx \sum_{i=1}^B p_i \Delta x = 1, \qquad p_i = \frac{N_i}{N_{\text{total}} \cdot \Delta x}
\f]

```cpp
num::Histogram histogram(100, -1.0, 1.0); // 100 uniform bins over [-1, 1)

for (double sample : samples) {
    histogram.fill(sample);
}

std::vector<double> pdf = histogram.pdf(); // Normalized probability density integrating to 1.0
```

---

---

## 3. Compile-Time Concepts

```cpp
static_assert(num::StreamingAccumulator<num::RunningStats>);
static_assert(num::MomentAccumulator<num::RunningStats>);
```

`MomentAccumulator` reports the first two moments through Welford's recurrence:

\f[
\mu_n = \mu_{n-1} + \frac{x_n - \mu_{n-1}}{n}, \qquad s^2 = \frac{M_{2,n}}{n - 1}
\f]

The recurrence is used rather than \f$\sum x^2 - n\mu^2\f$ because the latter cancels catastrophically when the variance is small relative to the mean.
