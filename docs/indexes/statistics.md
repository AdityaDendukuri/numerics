# Statistics and sampling

Streaming statistics, selection, random engines and Monte Carlo sampling. The concepts are on the [Concepts](concepts.md) page.

## Running statistics <stats/stats.hpp>
num::running_stats num::basic_running_stats num::histogram num::basic_histogram num::autocorr_time

## Probability vectors <stats/probability.hpp>
num::clip_and_normalize_nonnegative num::weighted_sum

## Selection <stats/selection.hpp>
num::argmax num::argsort num::smallest_indices num::filter num::group_by

## Random engines <stochastic/rng.hpp>
num::rng num::rng64 num::markov::make_rng num::markov::make_seeded_rng

## Categorical sampling <stochastic/categorical.hpp>
num::categorical_sampler num::sample_categorical

## Random probes <stochastic/probe.hpp>
num::rademacher_probe num::gaussian_probe num::hutchinson_row_mean_square

## Metropolis and umbrella sampling <stochastic/mcmc.hpp>
num::markov::metropolis_stats num::markov::umbrella_stats num::markov::umbrella_window

## Sweeps <stochastic/detail/mcmc_impl.hpp>
num::markov::metropolis_sweep num::markov::metropolis_sweep_prob num::markov::umbrella_sweep num::markov::umbrella_sweep_prob

## Boltzmann tables <stochastic/boltzmann_table.hpp>
num::markov::boltzmann_accept num::markov::make_boltzmann_table
