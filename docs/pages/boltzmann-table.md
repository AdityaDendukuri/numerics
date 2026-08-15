# Boltzmann Acceptance Table {#page_boltzmann_table}

`include/stochastic/boltzmann_table.hpp` provides two small utilities in
`num::markov` for computing Metropolis acceptance probabilities.

## Definition

Every Metropolis sweep must decide whether to accept a proposed spin flip.
The acceptance probability is \f$\min(1,\, e^{-\beta \Delta E})\f$.

In a typical Ising sweep over \f$N^2 = 90\,000\f$ sites this is evaluated millions
of times per second.  Calling `std::exp` at runtime is avoidable because \f$\Delta E\f$
is discrete: for the 2D Ising model,

\f[
\Delta E = 2J s \cdot \sum_{\text{nbrs}} s_j - 2F s,
\qquad s,\,s_j \in \{-1,+1\},\; \sum_{\text{nbrs}} \in \{-4,-2,0,2,4\}
\f]

so only 10 distinct values of \f$\Delta E\f$ occur.  Pre-computing a
\f$2 \times 5\f$ lookup table eliminates `exp` from the hot path entirely.

## Routine Reference

```cpp
namespace num::markov {

// Single evaluation: min(1, exp(-beta DeltaE))
double boltzmann_accept(double dE, double beta) noexcept;

// Precompute a table for a discrete DeltaE set
std::vector<double> make_boltzmann_table(const std::vector<double>& dEs, double beta);

}
```

## Examples

### Single Probability

```cpp
double p = num::markov::boltzmann_accept(dE, beta);
```

### Flat Lookup Table

```cpp
std::vector<double> dEs = {-4.0, -2.0, 0.0, 2.0, 4.0};
auto table = num::markov::make_boltzmann_table(dEs, beta);

auto probability = [&](double dE) {
    const int slot = static_cast<int>((dE + 4.0) / 2.0);
    return table[static_cast<std::size_t>(slot)];
};
```
