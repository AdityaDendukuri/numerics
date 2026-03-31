/// @file apps/tdse/main.cpp
/// @brief 2D time-dependent Schrodinger equation -- double slit.
///
/// Gaussian wavepacket, DoubleSlit potential, N=256, Strang splitting.
/// Phase-HSV colormap: hue = phase angle, brightness = probability amplitude.
/// SPACE pause  R reset  +/- substeps

#include "numerics.hpp"
#include "viz/viz.hpp"
#include "tdse/sim.hpp"
#include <algorithm>
#include <cmath>

// Compile-time alias (zero runtime overhead):
//   static_cast<float>(x)  →  cast<float>(x)
template<class To, class From>
constexpr To cast(From x) {
    return static_cast<To>(x);
}

static constexpr int    N   = 256;
static constexpr double L   = 10.0;
static constexpr double DT  = 0.004;
static constexpr int    WIN = 900;

static void init(tdse::TDSESolver& solver) {
    solver.set_potential(tdse::Potential::DoubleSlit);
    solver.init_gaussian(L * 0.2, L * 0.5, 5.0, 0.0, L * 0.06);
}

int main() {
    tdse::TDSESolver solver(N, L, DT);
    init(solver);

    num::viz::run("TDSE: Double Slit", WIN, WIN, [&](num::viz::Frame& f) {
        if (f.reset_pressed())
            init(solver);

        f.step([&] { solver.step(); });

        double max_prob = 1e-20;
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                max_prob = std::max(max_prob, solver.prob(i, j));

        f.field(N, [&](int col, int row) {
            return num::viz::phase_hsv_color(solver.prob(col, row),
                                             solver.phase_ang(col, row),
                                             max_prob);
        });
    });
}
