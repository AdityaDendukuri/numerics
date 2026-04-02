/// @file examples/ising_demo.cpp
/// @brief Ising model magnetization curve via num::solve + Metropolis.
///
/// Sweeps temperature from T=0.5 to T=4.0 and records |<m>| at each point.
/// The spontaneous-symmetry-breaking phase transition at Tc ~ 2.27 is visible
/// as the magnetization drops sharply from ~1 to ~0.

#include "numerics.hpp"
#include "spatial/pbc_lattice.hpp"
#include "stochastic/boltzmann_table.hpp"
#include <cmath>
#include <numeric>
#include <random>

int main() {
    const int N                    = 64;
    const int NN                   = N * N;
    const int equilibration_sweeps = 2000;
    const int measurement_sweeps   = 500;

    num::PBCLattice2D   nbr(N);
    std::vector<double> spins(NN);
    std::mt19937        rng(42);

    num::Series curve;
    for (double T = 0.5; T <= 4.01; T += 0.1) {
        std::fill(spins.begin(), spins.end(), 1.0);
        double beta = 1.0 / T;

        auto accept = [&](int i) {
            double ns = spins[nbr.up[i]] + spins[nbr.dn[i]]
                      + spins[nbr.lt[i]] + spins[nbr.rt[i]];
            return num::markov::boltzmann_accept(2.0 * spins[i] * ns, beta);
        };
        auto flip    = [&](int i) { spins[i] = -spins[i]; };
        auto measure = [&]() {
            return std::abs(std::accumulate(spins.begin(), spins.end(), 0.0) / NN);
        };

        double m = num::solve(
            num::MCMCProblem{accept, flip, NN},
            num::Metropolis{.equilibration=equilibration_sweeps,
                            .measurements=measurement_sweeps},
            measure, rng);

        curve.store(T, m);
    }

    num::plt::plot(curve);
    num::plt::title("Ising model: magnetization vs temperature (N=64)");
    num::plt::xlabel("T");
    num::plt::ylabel("|<m>|");
    num::plt::savefig("ising_demo.png");
}
