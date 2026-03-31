/// @file include/tdse/sim.hpp
/// @brief 2-D Time-Dependent Schrodinger Equation solver
///
/// Algorithm: Strang operator splitting
///   e^{-iHdt} ~= e^{-iVdt/2} * e^{-iTx*dt/2} * e^{-iTy*dt} * e^{-iTx*dt/2} *
///   e^{-iVdt/2}
///
/// Each kinetic sub-step uses Crank-Nicolson with a complex tridiagonal solve
/// (Thomas algorithm). Potential kick is a diagonal phase multiplication: psi
/// *= exp(-i*V*tau).
///
/// Eigenstate computation uses num::lanczos on the real Hamiltonian matrix (H
/// is real symmetric for real V, so eigenstates are real).
///
/// Bessel function zeros (for CircularWell exact eigenvalues) are found via
/// num::brent. Norm/energy observables use num::gauss_legendre on radial
/// marginals.
///
/// Grid: NxN interior points, domain [0,L]x[0,L], Dirichlet BCs (psi=0 on
/// boundary). Storage: row-major  idx = i*N + j,  i = row (x), j = col (y).
#pragma once

#include "numerics.hpp"
#include <vector>
#include <complex>
#include <cmath>

namespace tdse {

using num::CVector;
using num::idx;
using num::real;
using num::Vector;

//  Potential types

enum class Potential {
    Free,         ///< V = 0 (free particle)
    Barrier,      ///< Single vertical barrier with one gap  -- tunnelling
    DoubleSlit,   ///< Double-slit wall  -- interference
    Harmonic,     ///< V = 1/2*omega^2*r^2  (coherent state dynamics)
    CircularWell, ///< V = 0 inside radius R, V = V0 outside (Bessel
                  ///< eigenstates)
};

inline const char* potential_name(Potential p) {
    switch (p) {
        case Potential::Free:
            return "Free particle";
        case Potential::Barrier:
            return "Single barrier";
        case Potential::DoubleSlit:
            return "Double slit";
        case Potential::Harmonic:
            return "Harmonic oscillator";
        case Potential::CircularWell:
            return "Circular well";
    }
    return "?";
}

//  Eigenmode (real, from Lanczos)

struct EigenMode {
    double              energy; ///< Ritz eigenvalue (~= energy)
    std::vector<double> phi;    ///< NxN real wavefunction, L^2-normalised
    double              exact_energy =
        -1; ///< Analytical eigenvalue if available (CircularWell)
};

//  Stats

struct Stats {
    double step_ms = 0; ///< Wall time for one step()
    double norm    = 1; ///< integral|psi|^2 dA  (should stay ~= 1)
    double energy  = 0; ///< <H> = integralpsi* H psi dA
    int    n_modes = 0; ///< Number of computed eigenmodes
};

//  TDSESolver

class TDSESolver {
  public:
    int    N;         ///< Interior grid points per axis
    double L;         ///< Domain length ([0,L]x[0,L])
    double h;         ///< Grid spacing = L/(N+1)
    double dt;        ///< Time step

    num::CVector psi; ///< NxN wavefunction (complex, length N^2)
    num::Vector  V;   ///< NxN potential    (real,    length N^2)

    std::vector<EigenMode> modes;
    bool                   modes_ready = false;

    Stats stats;

    TDSESolver(int N, double L, double dt);

    /// Build potential and recompute Thomas factorisations.
    void set_potential(Potential p, double param = 0.0);

    /// Gaussian wavepacket: psi = A*exp(-(r-r0)^2/sigma^2)*exp(i*k*r), then
    /// normalised.
    void init_gaussian(double x0,
                       double y0,
                       double kx,
                       double ky,
                       double sigma);

    /// Set psi to the k-th eigenmode (modes must be computed first).
    void set_mode(int k);

    /// Advance one full time step (Strang splitting).
    void step();

    /// Run Lanczos to find the k lowest eigenstates of H.
    /// For CircularWell also fills EigenMode::exact_energy via Bessel zero
    /// search.
    void compute_modes(int k = 8);

    inline int at(int i, int j) const {
        return i * N + j;
    }

    double prob(int i, int j) const {
        const auto z = psi[at(i, j)];
        return z.real() * z.real() + z.imag() * z.imag();
    }
    double phase_ang(int i, int j) const {
        return std::arg(psi[at(i, j)]);
    }

    /// integral|psi|^2 dA  (Gauss-Legendre over each grid strip).
    double compute_norm() const;

    /// <psi|H|psi> using 5-point Laplacian and the potential.
    double compute_energy() const;

    void renormalize();

  private:
    void kick_V(double tau);  ///< psi *= exp(-i*V*tau)
    void sweep_x(double tau); ///< CN in x (col_fiber_sweep)
    void sweep_y(double tau); ///< CN in y (row_fiber_sweep)

    num::CrankNicolsonADI
        adi_;         ///< Prefactored CN tridiagonals for Strang splitting

    void refactor_(); ///< Rebuild ADI factorisations from h and dt

    Potential current_potential_ = Potential::Free;
};

} // namespace tdse
