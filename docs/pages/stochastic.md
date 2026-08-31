# Stochastic & MCMC Sampling {#page_stochastic}

Categorical sampling, Metropolis–Hastings sweeps, precomputed Boltzmann tables, and Umbrella sampling.

---

## 1. Categorical Sampling

Draws discrete states from unnormalized non-negative weights \f$w_k \ge 0\f$: \f$P(X = k) = \frac{w_k}{\sum_j w_j}\f$.

```cpp
#include <numerics.hpp>

auto rng = num::markov::make_rng(1234);

num::Vector weights{1.0, 2.0, 7.0}; // [10%, 20%, 70%]

// Single draw
num::idx state = num::sample_categorical(weights.span(), rng);

// Reusable sampler for repeated draws
num::CategoricalSampler sampler(weights.span());
num::idx draw = sampler(rng);
```

---

## 2. Metropolis–Hastings Sweeps

Generates Markov chains targeting the Boltzmann distribution \f$\pi(x) \propto e^{-\beta E(x)}\f$.

```cpp
auto delta_energy = [&](num::idx site) { return energy_change(spins, site); };
auto flip         = [&](num::idx site) { spins[site] = -spins[site]; };

num::markov::MetropolisStats stats = num::markov::metropolis_sweep(
    spins.size(), delta_energy, flip, beta, rng);

double acceptance_rate = stats.acceptance_rate();
```

### Precomputed Boltzmann Table (num::markov::make_boltzmann_table)
Avoids runtime `std::exp` calls when \f$\Delta E\f$ takes values in a known discrete set:

```cpp
std::vector<double> discrete_dE{-8.0, -4.0, 0.0, 4.0, 8.0};
auto table = num::markov::make_boltzmann_table(discrete_dE, beta);

auto stats = num::markov::metropolis_sweep_prob(
    spins.size(),
    [&](num::idx site) { return table[slot_for(spins, site)]; },
    flip,
    rng);
```

---

## 3. Umbrella Sampling (num::markov::umbrella_sweep)

Constrains Markov walks to an order parameter window \f$\xi(x) \in [\xi_{\text{lo}}, \xi_{\text{hi}}]\f$:

```cpp
num::markov::UmbrellaWindow window{.lo = 40, .hi = 60};

auto save          = [&] { saved_spins = spins; };
auto restore       = [&] { spins = saved_spins; };
auto measure_order = [&] { return count_magnetization(spins); };

auto res = num::markov::umbrella_sweep(
    spins.size(), delta_energy, flip, save, restore, measure_order, window, beta, rng);
// res.reverted is true if proposal moved outside [lo, hi]
```

---

## 4. Concepts

```cpp
static_assert(num::RandomEngine<std::mt19937>);
static_assert(!num::RandomEngine<double>);
static_assert(num::CategoricalSampling<num::CategoricalSampler, std::mt19937>);
```

---

## Complete Example

@example 09_mcmc_bayesian_sampling.cpp

