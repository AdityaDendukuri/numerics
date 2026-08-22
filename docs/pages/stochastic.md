# Stochastic Examples {#page_stochastic}

## Reproducible Random Engine

```cpp
#include <numerics.hpp>

auto rng = num::markov::make_rng(1234); // Same seed produces the same stream.
```

## Entropy-Seeded Engine

```cpp
auto rng = num::markov::make_seeded_rng(); // Seed from std::random_device.
```

Both helpers default to `std::mt19937`; pass another engine as the template argument.

## One Categorical Draw

```cpp
num::Vector weights{1.0, 2.0, 7.0}; // Weights need not sum to one.
num::idx state = num::sample_categorical(weights.span(), rng);
```

```cpp
num::Vector invalid{-1.0, 2.0};
num::sample_categorical(invalid.span(), rng); // Throws: weights must be nonnegative.
```

## Repeated Categorical Draws

```cpp
num::CategoricalSampler sampler(weights.span()); // Precompute a fixed distribution.

num::idx first = sampler(rng);
num::idx second = sampler(rng);
```

## Boltzmann Acceptance

```cpp
double probability = num::markov::boltzmann_accept(delta_energy, beta); // One probability.
```

```cpp
std::vector<double> energies{-8.0, -4.0, 0.0, 4.0, 8.0};
auto table = num::markov::make_boltzmann_table(energies, beta); // Reuse discrete probabilities.
```

## Metropolis Energy Sweep

```cpp
auto delta_energy = [&](num::idx site) { return energy_change(spins, site); };
auto flip = [&](num::idx site) { spins[site] = -spins[site]; };

num::markov::MetropolisStats stats = num::markov::metropolis_sweep(
    spins.size(), delta_energy, flip, beta, rng); // Attempt one move per site.
```

## Metropolis Probability Sweep

```cpp
auto acceptance = [&](num::idx site) {
    return table[energy_slot(spins, site)]; // Skip exp() in the sweep.
};

auto stats = num::markov::metropolis_sweep_prob(
    spins.size(), acceptance, flip, rng);
```

## Acceptance Metadata

```cpp
num::idx accepted = stats.accepted;
num::idx attempted = stats.total;
double rate = stats.acceptance_rate(); // Zero when no moves were attempted.
```

## Umbrella Window

```cpp
num::markov::UmbrellaWindow window{.lo = 40, .hi = 60}; // Inclusive bounds.
bool inside = window.contains(order_parameter());
```

## Umbrella Sweep

```cpp
auto save = [&] { saved_spins = spins; };
auto restore = [&] { spins = saved_spins; };
auto measure_order = [&] { return count_up_spins(spins); };

auto result = num::markov::umbrella_sweep(
    spins.size(), delta_energy, flip, save, restore, measure_order, window, beta, rng);
```

```cpp
bool rolled_back = result.reverted;             // True when the proposal left the window.
num::idx order = result.order_param;             // Order after any rollback.
double rate = result.mc.acceptance_rate();       // Inner Metropolis acceptance.
```

`umbrella_sweep_prob` accepts precomputed probabilities in place of energy changes.

## Model-Level Sampling

```cpp
num::MCMCModel model{
    .accept_prob = acceptance,
    .propose = flip,
    .n_sites = static_cast<int>(spins.size()),
    .measure = [&] { return magnetization(spins); },
};

num::Metropolis method{.equilibration = 1000, .measurements = 5000};
double mean_magnetization = num::sample(model, method, rng);
```

## Running Statistics

```cpp
num::RunningStats stats;

for (double observation : samples) {
    stats.update(observation); // Update without storing prior observations.
}
```

```cpp
double mean = stats.mean;
double variance = stats.variance();       // Unbiased sample variance.
double deviation = stats.std_dev();
double mean_error = stats.stderr_mean();  // Assumes uncorrelated observations.
```

```cpp
stats.reset(); // Restore the empty accumulator.
```

## Histogram Construction

```cpp
num::Histogram histogram(100, -1.0, 1.0); // 100 bins over [-1, 1).

histogram.fill(0.25);
histogram.fill(0.25, 2.0); // Weighted observation.
```

## Histogram Queries

```cpp
num::idx bin = histogram.bin(0.25);
double centre = histogram.bin_centre(bin);
double width = histogram.bin_width();
double weight = histogram.total();
std::vector<double> density = histogram.pdf(); // Integrates to one when nonempty.
```

```cpp
histogram.reset(); // Zero all bin counts.
```

## Autocorrelation Time

```cpp
double tau = num::autocorr_time(samples.data(), samples.size()); // Automatic windowing.
```

## Complete Program

@example 09_mcmc_bayesian_sampling.cpp
