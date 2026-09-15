# Stochastic & MCMC Sampling {#page_stochastic}

Random engines, categorical sampling, Rademacher probes, Metropolis–Hastings sweeps, precomputed Boltzmann tables, and Umbrella sampling.

---

## 0. Random Engines

`num::rng` and `num::rng64` name the library's default engines (`std::mt19937`
and `std::mt19937_64`). They are plain aliases, so any standard distribution or
generic `RNG` parameter accepts them unchanged; naming them keeps the choice of
engine in one place.

```cpp
#include <numerics.hpp>

num::rng   generator(42);                 // fixed seed
num::rng64 wide = num::markov::make_rng<num::rng64>(42);
auto       entropy = num::markov::make_seeded_rng(); // num::rng from std::random_device
```

---

## 1. Categorical Sampling

Draws discrete states from unnormalized non-negative weights \f$w_k \ge 0\f$: \f$P(X = k) = \frac{w_k}{\sum_j w_j}\f$.

```cpp
#include <numerics.hpp>

auto rng = num::markov::make_rng(1234);

num::vec weights{1.0, 2.0, 7.0}; // [10%, 20%, 70%]

// Single draw
num::idx state = num::sample_categorical(weights.span(), rng);

// Reusable sampler for repeated draws
num::categorical_sampler sampler(weights.span());
num::idx draw = sampler(rng);
```

---

## 1b. Rademacher Probes

`num::rademacher_probe(n, p, rng)` returns an \f$n \times p\f$ matrix whose
columns are independent \f$\pm 1\f$ vectors \f$z_k\f$, the input to Hutchinson
estimators. If column \f$k\f$ of `probed` holds \f$B z_k\f$, then
`num::hutchinson_row_mean_square(probed)` estimates \f$\operatorname{diag}(B B^T)\f$.

```cpp
num::rng generator(7);
const num::mat probe = num::rademacher_probe(n, 64, generator);
num::mat probed(n, 64, 0.0);
for (num::idx k = 0; k < 64; ++k) {
    num::vec z(n, 0.0), Bz(n, 0.0);
    for (num::idx j = 0; j < n; ++j) z[j] = probe(j, k);
    B.apply(z, Bz);
    for (num::idx j = 0; j < n; ++j) probed(j, k) = Bz[j];
}
const num::vec diagonal_estimate = num::hutchinson_row_mean_square(probed);
```

---

## 2. Metropolis–Hastings Sweeps

Generates Markov chains targeting the Boltzmann distribution \f$\pi(x) \propto e^{-\beta E(x)}\f$.

```cpp
auto delta_energy = [&](num::idx site) { return energy_change(spins, site); };
auto flip         = [&](num::idx site) { spins[site] = -spins[site]; };

num::markov::metropolis_stats stats = num::markov::metropolis_sweep(
    spins.size(), delta_energy, flip, beta, rng);

double acceptance_rate = stats.acceptance_rate();
```

### Precomputed Boltzmann Table (num::markov::make_boltzmann_table)
Avoids runtime `std::exp` calls when \f$\Delta E\f$ takes values in a known discrete set:

```cpp
const double beta = 1.0;   // inverse temperature; `num::beta` is the beta function
num::array<double> discrete_dE{-8.0, -4.0, 0.0, 4.0, 8.0};
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
num::markov::umbrella_window window{.lo = 40, .hi = 60};

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
static_assert(std::uniform_random_bit_generator<num::rng>);
static_assert(!std::uniform_random_bit_generator<double>);
static_assert(num::categorical_sampling<num::categorical_sampler, num::rng>);
```

---

## Complete Example

@example 09_mcmc_bayesian_sampling.cpp

