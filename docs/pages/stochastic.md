# Stochastic Examples {#page_stochastic}

Stochastic routines cover random-number setup, Metropolis sweeps, Boltzmann
acceptance tables, online statistics, histograms, and autocorrelation estimates.

## Metropolis Driver

```cpp
#include <numerics.hpp>

std::vector<int> spin(N, 1);

auto accept = [&](int i) {
    const double dE = energy_change_for_flip(spin, i);
    return std::min(1.0, std::exp(-beta * dE));
};

auto propose = [&](int i) {
    spin[i] = -spin[i];
};

auto measure = [&]() {
    double m = 0.0;
    for (int s : spin) m += s;
    return m / static_cast<double>(N);
};

auto rng = num::make_seeded_rng(1234);
double mean_m = num::sample(
    num::MCMCModel{accept, propose, N, measure},
    num::Metropolis{.equilibration = 1000, .measurements = 5000},
    rng);
```

The model stores the acceptance, proposal, and measurement functions. The sampler
tag stores the equilibration and measurement counts. Sampling uses `sample()`,
the stochastic counterpart to `solve()`.

## Precomputed Boltzmann Probabilities

```cpp
auto table = num::markov::make_boltzmann_table({-8.0, -4.0, 0.0, 4.0, 8.0}, beta);

auto lookup = [&](double dE) {
    const int slot = static_cast<int>((dE + 8.0) / 4.0);
    return table[static_cast<std::size_t>(slot)];
};
```

Use this when \f$\Delta E\f$ takes values from a small discrete set.

## Running Statistics

```cpp
num::RunningStats stats;

for (double sample : samples) {
    stats.update(sample);
}

double mean = stats.mean;
double stderr = stats.stderr_mean();
```

Welford updates avoid storing the full sample stream.

## Histograms and Autocorrelation

```cpp
num::Histogram H(100, -1.0, 1.0);

for (double x : samples) {
    H.fill(x);
}

std::vector<double> pdf = H.pdf();
double tau = num::autocorr_time(samples.data(), samples.size());
```

`autocorr_time` estimates the integrated autocorrelation time from a stored
time series.
