# Boltzmann Acceptance Table {#page_boltzmann_table}

`include/stochastic/boltzmann_table.hpp` provides two small utilities in
`num::markov` for computing Metropolis acceptance probabilities.

---

## Motivation

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

---

## API

```cpp
namespace num::markov {

// Single evaluation: min(1, exp(-beta DeltaE))
double boltzmann_accept(double dE, double beta) noexcept;

// Precompute a table for a discrete DeltaE set
std::vector<double> make_boltzmann_table(const std::vector<double>& dEs, double beta);

}
```

`boltzmann_accept` replaces the inline ternary `(dE <= 0.0) ? 1.0 : std::exp(-beta*dE)`
that previously appeared directly in `rebuild_boltz()`.

---

## Usage

### Direct replacement in Ising

```cpp
// Before
boltz[si][ni] = (dE <= 0.0) ? 1.0 : std::exp(-beta * dE);

// After
boltz[si][ni] = num::markov::boltzmann_accept(dE, beta);
```

### Building a flat lookup table

```cpp
// For any solver with discrete energy differences:
std::vector<double> dEs = {-4.0, -2.0, 0.0, 2.0, 4.0};  // scaled by 2J
auto table = num::markov::make_boltzmann_table(dEs, beta);
// table[2] = boltzmann_accept(0.0, beta) = 1.0  (always accept)
// table[3] = boltzmann_accept(2.0, beta) = exp(-2*beta)
```

**Used by:** Ising `IsingLattice::rebuild_boltz`.
