/// @file examples/ising_demo.cpp
/// @brief Ising model magnetization curve via num::markov and num::PBCLattice2D.
///
/// Sweeps temperature from T=0.5 to T=4.0 and records |<m>| at each point.
/// The spontaneous-symmetry-breaking phase transition at Tc ~ 2.27 is visible
/// as the magnetization drops sharply from ~1 to ~0.
///
/// Starts from the ordered (all-up) state at each temperature so the system
/// equilibrates to the correct phase rather than getting trapped between ±M states.

#include "numerics.hpp"
#include "spatial/pbc_lattice.hpp"
#include "stochastic/mcmc.hpp"
#include "stochastic/boltzmann_table.hpp"
#include <cmath>
#include <random>
#include <numeric>

int main() {
    const int N  = 64;
    const int NN = N * N;
    const int equilibration_sweeps = 2000;
    const int measurement_sweeps   = 500;

    num::PBCLattice2D nbr(N);
    std::vector<double> spins(NN);
    std::mt19937 rng(42);

    num::Series curve;
    for (double T = 0.5; T <= 4.01; T += 0.1) {
        // Ordered start: all spins up.  At high T it disorders quickly;
        // at low T it stays in the correct ordered phase without tunneling.
        std::fill(spins.begin(), spins.end(), 1.0);

        double beta = 1.0 / T;

        auto acc_prob = [&](int i) {
            double ns = spins[nbr.up[i]] + spins[nbr.dn[i]]
                      + spins[nbr.lt[i]] + spins[nbr.rt[i]];
            return num::markov::boltzmann_accept(2.0 * spins[i] * ns, beta);
        };
        auto flip = [&](int i) { spins[i] = -spins[i]; };

        for (int s = 0; s < equilibration_sweeps; ++s)
            num::markov::metropolis_sweep_prob(NN, acc_prob, flip, rng);

        double m = 0.0;
        for (int s = 0; s < measurement_sweeps; ++s) {
            num::markov::metropolis_sweep_prob(NN, acc_prob, flip, rng);
            m += std::abs(std::accumulate(spins.begin(), spins.end(), 0.0) / NN);
        }
        curve.store(T, m / measurement_sweeps);
    }

    num::plt::plot(curve);
    num::plt::title("Ising model: magnetization vs temperature (N=64)");
    num::plt::xlabel("T");
    num::plt::ylabel("|<m>|");
    num::plt::savefig("ising_demo.png");
}
