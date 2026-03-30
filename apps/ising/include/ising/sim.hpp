/// @file include/ising/sim.hpp
/// @brief 2D Ising model on an NxN periodic lattice.
///
/// Wraps the numerics library's Metropolis and umbrella-sampling sweeps into a
/// self-contained IsingLattice.  No rendering code -- pure physics.
///
/// Two sweep modes:
///   sweep()            -- classic Metropolis (free evolution)
///   sweep_umbrella()   -- umbrella sampling constrained to a nucleus-size window
///
/// Observable accessors:
///   magnetization()    -- |<m>| via SIMD dot product
///   energy_per_spin()  -- <E>/N^2 via sparse adjacency matvec
///   mean_field_m()     -- static mean-field order parameter (num::newton)
#pragma once

#include "numerics.hpp"
#include "stats/stats.hpp"
#include "analysis/roots.hpp"
#include "stochastic/markov.hpp"
#include "spatial/pbc_lattice.hpp"
#include "spatial/connected_components.hpp"

#include <random>
#include <vector>
#include <cmath>

using num::real;
using num::idx;

// IsingLattice

template<int N>
struct IsingLattice {
    static constexpr int  NN  = N * N;
    static constexpr real J   = 1.0;
    static const     real TC;   ///< Onsager exact critical temperature

    num::Vector  spins;    ///< NN spins: values +/-1
    real         beta;     ///< Inverse temperature
    real         F;        ///< External field

    IsingLattice()
        : spins(NN, 1.0)
        , beta(1.0 / (1.0 / 0.58))   // T ~ 1.724  (matches IsingNucleation defaults)
        , F(0.0)
        , ones_(NN, 1.0)
        , nbr_buf_(NN, 0.0)
        , adj_(build_adj())
        , nbr_(N)
        , save_buf_(NN)
        , rng_(std::random_device{}())
        , dist_(0.0, 1.0)
    {
        rebuild_boltz();
        random_init();
    }

    // Configuration

    void set_temperature(real T) { beta = (T > 1e-6) ? 1.0/T : 1e6; rebuild_boltz(); }
    void set_field(real Fnew)    { F = Fnew; rebuild_boltz(); }
    real temperature() const     { return (beta > 1e-6) ? 1.0/beta : 1e6; }

    // Initial conditions

    void random_init() {
        for (int i = 0; i < NN; ++i)
            spins[i] = (dist_(rng_) < 0.5) ? 1.0 : -1.0;
    }

    void all_up() {
        for (int i = 0; i < NN; ++i) spins[i] = 1.0;
    }

    /// Square spin-down nucleus of area ~r centred at the middle of the lattice.
    void generate_nucleus(real r = 100.0) {
        all_up();
        int half = static_cast<int>(std::sqrt(r) * 0.5);
        int cx   = N / 2;
        for (int row = cx - half; row < cx + half; ++row)
            for (int col = cx - half; col < cx + half; ++col)
                spins[row * N + col] = -1.0;
    }

    // Sweeps

    /// Standard Metropolis sweep: N^2 random spin picks with replacement.
    /// Uses a precomputed Boltzmann table -- no exp() at runtime.
    void sweep() {
        num::markov::metropolis_sweep_prob(
            static_cast<idx>(NN),
            [&](idx i) {
                real s  = spins[i];
                real ns = spins[nbr_.up[i]] + spins[nbr_.dn[i]]
                        + spins[nbr_.lt[i]] + spins[nbr_.rt[i]];
                return boltz_[spin_idx(s)][nbr_idx(ns)];
            },
            [&](idx i) { spins[i] = -spins[i]; },
            rng_);
    }

    /// Umbrella sweep: revert if nucleus size leaves [lo, hi].
    /// Fills det with the connected-component result for rendering.
    int sweep_umbrella(int lo, int hi, num::ClusterResult& det) {
        auto result = num::markov::umbrella_sweep_prob(
            static_cast<idx>(NN),
            [&](idx i) {
                real s  = spins[i];
                real ns = spins[nbr_.up[i]] + spins[nbr_.dn[i]]
                        + spins[nbr_.lt[i]] + spins[nbr_.rt[i]];
                return boltz_[spin_idx(s)][nbr_idx(ns)];
            },
            [&](idx i) { spins[i] = -spins[i]; },
            [&]() { for (int i = 0; i < NN; ++i) save_buf_[i] = spins[i]; },
            [&]() { for (int i = 0; i < NN; ++i) spins[i] = save_buf_[i]; },
            [&]() -> idx {
                det = num::connected_components(NN,
                    [&](int i) { return spins[i] < 0.0; },
                    [&](int i, auto&& visit) {
                        visit(nbr_.up[i]); visit(nbr_.dn[i]);
                        visit(nbr_.lt[i]); visit(nbr_.rt[i]);
                    });
                return static_cast<idx>(det.largest_size);
            },
            num::markov::UmbrellaWindow{static_cast<idx>(lo), static_cast<idx>(hi)},
            rng_);
        return static_cast<int>(result.order_param);
    }

    // Observables

    /// |<m>| = |sum(s_i)| / N^2 via SIMD dot product.
    real magnetization() const {
        return std::abs(dot(spins, ones_, num::Backend::seq)) / static_cast<real>(NN);
    }

    /// <E>/N^2 using the sparse adjacency matrix and SIMD sparse_matvec.
    real energy_per_spin() {
        num::sparse_matvec(adj_, spins, nbr_buf_);
        real coup = -J * dot(spins, nbr_buf_, num::Backend::seq) / (4.0 * NN);
        real fld  = -F * dot(spins, ones_,    num::Backend::seq) / static_cast<real>(NN);
        return coup + fld;
    }

    /// Mean-field order parameter: solve m = tanh(4*beta*J*m + beta*F) via num::newton().
    static real mean_field_m(real beta_val, real F_val) {
        auto f  = [=](real m) {
            return std::tanh(4.0 * beta_val * J * m + beta_val * F_val) - m;
        };
        auto df = [=](real m) {
            real th = std::tanh(4.0 * beta_val * J * m + beta_val * F_val);
            return 4.0 * beta_val * J * (1.0 - th * th) - 1.0;
        };
        auto res = num::newton(f, df, 0.9);
        return std::max(-1.0, std::min(1.0, res.root));
    }

private:
    num::Vector       ones_;
    num::Vector       nbr_buf_;
    num::SparseMatrix adj_;
    num::PBCLattice2D nbr_;
    std::vector<real> save_buf_;
    std::mt19937      rng_;
    std::uniform_real_distribution<real> dist_;
    real              boltz_[2][5]{};

    // Boltzmann acceptance table: boltz_[spin_idx][nbr_idx] = min(1, exp(-beta*dE))
    // spin_idx: 0 = s=-1, 1 = s=+1
    // nbr_idx:  (nbr_sum/2 + 2) in {0..4}
    void rebuild_boltz() {
        for (int si = 0; si < 2; ++si) {
            real s = (si == 0) ? -1.0 : 1.0;
            for (int ni = 0; ni < 5; ++ni) {
                real ns = -4.0 + 2.0 * ni;
                real dE = 2.0 * J * s * ns - 2.0 * F * s;
                boltz_[si][ni] = num::markov::boltzmann_accept(dE, beta);
            }
        }
    }

    static int spin_idx(real s) { return s < 0.0 ? 0 : 1; }
    static int nbr_idx(real ns) { return static_cast<int>(ns * 0.5 + 2.0); }

    static num::SparseMatrix build_adj() {
        std::vector<idx>  rows_t, cols_t;
        std::vector<real> vals_t;
        rows_t.reserve(4 * NN);  cols_t.reserve(4 * NN);  vals_t.reserve(4 * NN);
        for (int row = 0; row < N; ++row) {
            for (int col = 0; col < N; ++col) {
                int i = row * N + col;
                int nbrs[4] = {
                    ((row-1+N)%N)*N + col,  ((row+1)%N)*N   + col,
                     row*N + (col-1+N)%N,    row*N + (col+1)%N
                };
                for (int nb : nbrs) {
                    rows_t.push_back(static_cast<idx>(i));
                    cols_t.push_back(static_cast<idx>(nb));
                    vals_t.push_back(1.0);
                }
            }
        }
        return num::SparseMatrix::from_triplets(NN, NN, rows_t, cols_t, vals_t);
    }
};

template<int N>
const real IsingLattice<N>::TC = 2.26918531421;
