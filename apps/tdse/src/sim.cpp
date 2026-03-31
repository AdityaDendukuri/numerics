/// @file src/sim.cpp
/// @brief Implementation of TDSESolver -- 2-D TDSE via Strang splitting +
/// Thomas algorithm.
///
/// Algorithm chain per step():
///   1. kick_V(dt/2)    -- diagonal phase multiplication
///   2. sweep_x(dt/2)   -- Crank-Nicolson in x (Thomas per column fiber)
///   3. sweep_y(dt)     -- Crank-Nicolson in y (Thomas per row fiber)
///   4. sweep_x(dt/2)   -- again
///   5. kick_V(dt/2)    -- again
///
/// Eigenstates: num::lanczos on the real N^2xN^2 Hamiltonian (matrix-free
/// matvec). Bessel zeros: num::brent on J_m(x) for exact CircularWell
/// eigenvalues.

#include "tdse/sim.hpp"

#include <chrono>
#include <stdexcept>
#include <cmath>

using namespace num;

namespace tdse {

using cplx = std::complex<double>;

TDSESolver::TDSESolver(int N_, double L_, double dt_)
    : N(N_)
    , L(L_)
    , h(L_ / static_cast<double>(N_ + 1))
    , dt(dt_)
    , psi(to_idx(N_ * N_), cplx(0.0, 0.0))
    , V(to_idx(N_ * N_), 0.0) {
    refactor_();
    init_gaussian(L_ * 0.25, L_ * 0.5, 4.0, 0.0, L_ * 0.07);
}

void TDSESolver::refactor_() {
    adi_ = CrankNicolsonADI(N, dt, h);
}

//  Potential setup

void TDSESolver::set_potential(Potential p, double param) {
    current_potential_ = p;
    scale(V, 0.0); // zero the potential before rebuilding

    const double cx = L * 0.5, cy = L * 0.5;
    const double V0 = 5000.0; // hard wall height

    if (p == Potential::Barrier) {
        // Vertical wall at x = L*0.55, gap centred at y = L/2, width = L*0.12
        double wall_x  = L * 0.55;
        double gap_cy  = cy;
        double gap_w   = L * 0.12;
        int    wall_th = std::max(1, N / 60); // ~4 cells thick

        int ix0 = static_cast<int>((wall_x / L) * N - wall_th);
        int ix1 = static_cast<int>((wall_x / L) * N + wall_th);

        for (int i = ix0; i <= std::min(ix1, N - 1); ++i) {
            for (int j = 0; j < N; ++j) {
                double y = (j + 1.0) * h;
                if (std::abs(y - gap_cy) > gap_w * 0.5)
                    V[at(i, j)] = V0;
            }
        }
    } else if (p == Potential::DoubleSlit) {
        double wall_x  = L * 0.5;
        double gap_sep = L * 0.12; // centre-to-centre separation
        double gap_w   = L * 0.06; // each slit width
        int    wall_th = std::max(1, N / 60);

        int ix0 = static_cast<int>((wall_x / L) * N - wall_th);
        int ix1 = static_cast<int>((wall_x / L) * N + wall_th);

        for (int i = ix0; i <= std::min(ix1, N - 1); ++i) {
            for (int j = 0; j < N; ++j) {
                double y        = (j + 1.0) * h;
                bool   in_slit1 = std::abs(y - (cy - gap_sep * 0.5))
                                < gap_w * 0.5;
                bool in_slit2 = std::abs(y - (cy + gap_sep * 0.5))
                                < gap_w * 0.5;
                if (!in_slit1 && !in_slit2)
                    V[at(i, j)] = V0;
            }
        }
    } else if (p == Potential::Harmonic) {
        double omega = (param > 0) ? param : 1.5;
        for (int i = 0; i < N; ++i) {
            double x = (i + 1.0) * h - cx;
            for (int j = 0; j < N; ++j) {
                double y    = (j + 1.0) * h - cy;
                V[at(i, j)] = 0.5 * omega * omega * (x * x + y * y);
            }
        }
    } else if (p == Potential::CircularWell) {
        double R = (param > 0) ? param : L * 0.4;
        for (int i = 0; i < N; ++i) {
            double x = (i + 1.0) * h - cx;
            for (int j = 0; j < N; ++j) {
                double y = (j + 1.0) * h - cy;
                if (std::sqrt(x * x + y * y) > R)
                    V[at(i, j)] = V0;
            }
        }
    }
    // Free: all zeros already set

    modes_ready = false;
}

//  Initial conditions

void TDSESolver::init_gaussian(double x0,
                               double y0,
                               double kx,
                               double ky,
                               double sigma) {
    for (int i = 0; i < N; ++i) {
        double x = (i + 1.0) * h;
        for (int j = 0; j < N; ++j) {
            double y  = (j + 1.0) * h;
            double dx = x - x0, dy = y - y0;
            double env = std::exp(-(dx * dx + dy * dy) / (2.0 * sigma * sigma));
            double phi = kx * dx + ky * dy;
            psi[at(i, j)] = env * cplx(std::cos(phi), std::sin(phi));
        }
    }
    renormalize();
}

void TDSESolver::set_mode(int k) {
    if (!modes_ready || k < 0 || k >= static_cast<int>(modes.size()))
        throw std::runtime_error("tdse: modes not ready or k out of range");
    const auto& phi = modes[k].phi;
    for (int n2 = 0; n2 < N * N; ++n2)
        psi[n2] = cplx(phi[n2], 0.0);
    // already normalised
}

//  Strang splitting sub-steps

void TDSESolver::kick_V(double tau) {
    for (int n2 = 0; n2 < N * N; ++n2) {
        double angle = -V[n2] * tau;
        psi[n2] *= cplx(std::cos(angle), std::sin(angle));
    }
}

void TDSESolver::sweep_x(double tau) {
    adi_.sweep(psi, true, tau);
}
void TDSESolver::sweep_y(double tau) {
    adi_.sweep(psi, false, tau);
}

//  step()  -- Strang splitting

void TDSESolver::step() {
    auto t0 = std::chrono::high_resolution_clock::now();

    kick_V(dt * 0.5);
    sweep_x(dt * 0.5);
    sweep_y(dt);
    sweep_x(dt * 0.5);
    kick_V(dt * 0.5);

    auto t1       = std::chrono::high_resolution_clock::now();
    stats.step_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
}

//  Observables

double TDSESolver::compute_norm() const {
    // ||psi||_L2 = norm(psi) * h  (rectangle rule on uniform grid)
    return norm(psi) * h;
}

double TDSESolver::compute_energy() const {
    // <H> = <psi | -1/(2h^2) * Lap + V | psi> * h^2
    // laplacian_stencil_2d gives lap[k] = (sum of 4 nbrs) - 4*psi[k]
    // (Dirichlet)
    CVector lap(psi.size());
    laplacian_stencil_2d(psi, lap, N);

    const double c = 1.0 / (2.0 * h * h);
    double       E = 0.0;
    for (int k = 0; k < N * N; ++k) {
        cplx Hpsi = -c * lap[k] + cplx(V[k]) * psi[k];
        E += psi[k].real() * Hpsi.real() + psi[k].imag() * Hpsi.imag();
    }
    return E * h * h;
}

void TDSESolver::renormalize() {
    double inv = 1.0 / (norm(psi) * h);
    scale(psi, cplx{inv, 0.0});
}

//  Eigenstate computation  -- Lanczos on the real Hamiltonian

void TDSESolver::compute_modes(int k) {
    // H is real symmetric -> eigenstates are real. Work entirely in real
    // arithmetic.
    const int    n2 = N * N;
    const double c  = 1.0 / (2.0 * h * h);

    // Matrix-free Hamiltonian matvec: H = -c * Lap + V  (Dirichlet)
    Vector lap_buf(n2);
    auto   ham_mv = [&](const Vector& x, Vector& y) {
        laplacian_stencil_2d(x, lap_buf, N);
        for (int i = 0; i < n2; ++i)
            y[i] = -c * lap_buf[i] + V[i] * x[i];
    };

    idx  max_steps = std::min(to_idx(5 * k), to_idx(n2));
    auto res       = lanczos(ham_mv, to_idx(n2), k, 1e-8, max_steps);

    modes.clear();
    int got = static_cast<int>(res.ritz_values.size());
    modes.reserve(got);

    for (int m = 0; m < got; ++m) {
        EigenMode em;
        em.energy = res.ritz_values[m];
        em.phi.resize(n2);

        double norm_sq = 0;
        for (int r = 0; r < n2; ++r) {
            em.phi[r] = res.ritz_vectors(r, m);
            norm_sq += em.phi[r] * em.phi[r];
        }
        double inv = 1.0 / std::sqrt(norm_sq * h * h);
        for (double& v : em.phi)
            v *= inv;

        modes.push_back(std::move(em));
    }

    if (current_potential_ == Potential::CircularWell) {
        double R = L * 0.4; // default radius (matches set_potential)
        // Exact energies: E_{m,n} = j_{m,n}^2 / (2R^2)
        int mode_idx = 0;
        for (int m = 0; m <= 2 && mode_idx < got; ++m) {
            double bracket_lo = 1.0;
            for (int nz = 1; nz <= 3 && mode_idx < got; ++nz) {
                auto Jm = [m](real x) {
                    return bessel_j(m, x);
                };
                double lo = bracket_lo, hi = lo + 0.5;
                int    scan = 0;
                while (Jm(lo) * Jm(hi) > 0 && scan < 200) {
                    lo = hi;
                    hi += 0.5;
                    ++scan;
                }
                if (Jm(lo) * Jm(hi) > 0)
                    break;

                auto   res2    = brent(Jm, lo, hi, 1e-10);
                double E_exact = (res2.root * res2.root) / (2.0 * R * R);
                bracket_lo     = hi + 0.01;

                modes[mode_idx++].exact_energy = E_exact;
                if (m > 0 && mode_idx < got)
                    modes[mode_idx++].exact_energy = E_exact;
            }
        }
    }

    stats.n_modes = got;
    modes_ready   = true;
}

} // namespace tdse
