/// @file markov/backends/omp/markov.cpp
/// @brief OpenMP Markov chain Monte Carlo backend.
///
/// Target: parallel tempering (replica exchange Monte Carlo).
/// Multiple independent chains at different inverse temperatures run in
/// parallel; adjacent replicas periodically attempt configuration swaps.
///
/// TODO: implement parallel_tempering():
///   - Allocate N_replicas independent system states and RNGs
///   - #pragma omp parallel for over replicas (each runs metropolis_sweep_prob)
///   - After each block of sweeps, attempt swap between replica i and i+1:
///       dE_swap = (beta[i+1] - beta[i]) * (E[i] - E[i+1])
///       accept if u01 < exp(dE_swap)  (detailed balance is maintained)

namespace num::markov::backends::omp {
// (TODO: parallel tempering implementation)
} // namespace num::markov::backends::omp
