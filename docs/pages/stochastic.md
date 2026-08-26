# Stochastic & MCMC Sampling {#page_stochastic}

The `stochastic` module provides random number generators, categorical sampling, Metropolis–Hastings sweeps, Umbrella sampling, Welford running moments, and empirical density histograms.

---

## 1. Categorical & Discrete Sampling

Samples discrete states according to non-negative unnormalized weights \f$\mathbf{w} \in \mathbb{R}^K_{\ge 0}\f$:

\f[
P(X = k) = \frac{w_k}{\sum_{j=1}^K w_j}, \qquad k = 1, \dots, K
\f]

```cpp
#include <numerics.hpp>

auto rng = num::markov::make_rng(1234); // Seeded Mersenne Twister (std::mt19937)

num::Vector weights{1.0, 2.0, 7.0}; // Probability mass: [10%, 20%, 70%]
num::idx state = num::sample_categorical(weights.span(), rng);

// Reusable alias/CDF sampler for high-throughput discrete sampling
num::CategoricalSampler sampler(weights.span());
num::idx next_draw = sampler(rng);
```

---

## 2. Metropolis–Hastings MCMC Sweeps

Generates Markov chains converging to the canonical Boltzmann distribution \f$\pi(\mathbf{x}) \propto \exp(-\beta E(\mathbf{x}))\f$ with inverse temperature \f$\beta = \frac{1}{k_B T}\f$.

The Metropolis acceptance criterion guarantees detailed balance:

\f[
\alpha(\mathbf{x} \to \mathbf{x}') = \min\left(1, \exp(-\beta \Delta E)\right), \qquad \Delta E = E(\mathbf{x}') - E(\mathbf{x})
\f]

```cpp
auto delta_energy = [&](num::idx site) { return energy_change(spins, site); };
auto flip = [&](num::idx site) { spins[site] = -spins[site]; };

// Fast sweep evaluating delta_E
num::markov::MetropolisStats stats = num::markov::metropolis_sweep(
    spins.size(), delta_energy, flip, beta, rng);

double acceptance_ratio = stats.acceptance_rate(); // Accepted moves / Total proposals
```

### Precomputed Boltzmann Tables (Zero std::exp in Inner Loops)

When energy changes are restricted to discrete values \f$\Delta E \in \{\Delta E_1, \dots, \Delta E_M\}\f$ (e.g. Ising models):

```cpp
std::vector<double> energies{-8.0, -4.0, 0.0, 4.0, 8.0};
auto table = num::markov::make_boltzmann_table(energies, beta); // Precomputes exp(-beta * delta_E)

auto stats = num::markov::metropolis_sweep_prob(
    spins.size(), [&](num::idx site) { return table[energy_slot(spins, site)]; }, flip, rng);
```

---

## 3. Umbrella Sampling (Order Parameter Windows)

Constrains Markov walks to order parameter windows \f$\xi(\mathbf{x}) \in [\xi_{\text{lo}}, \xi_{\text{hi}}]\f$ to sample rare transitions and free energy barriers:

\f[
\pi_{\text{umbrella}}(\mathbf{x}) \propto \pi(\mathbf{x}) \cdot \mathbb{I}_{[\xi_{\text{lo}}, \xi_{\text{hi}}]}(\xi(\mathbf{x}))
\f]

```cpp
num::markov::UmbrellaWindow window{.lo = 40, .hi = 60};

auto save = [&] { saved_spins = spins; };
auto restore = [&] { spins = saved_spins; };
auto measure_order = [&] { return count_up_spins(spins); };

auto result = num::markov::umbrella_sweep(
    spins.size(), delta_energy, flip, save, restore, measure_order, window, beta, rng);

bool reverted = result.reverted; // True if proposal stepped outside the umbrella window
```

---

---

## Complete Example

@example 09_mcmc_bayesian_sampling.cpp

---

## Compile-Time Concepts

```cpp
static_assert(num::RandomEngine<std::mt19937>);
static_assert(!num::RandomEngine<double>);
static_assert(num::CategoricalSampling<num::CategoricalSampler, std::mt19937>);
```

Running statistics and histograms are documented on @ref page_stats.
