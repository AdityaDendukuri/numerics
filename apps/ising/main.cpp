/// @file apps/ising/main.cpp
/// @brief 2D Ising model -- interactive Metropolis with live sliders.
///
/// Adjust temperature T and external field h with the on-screen sliders.
/// The lattice evolves continuously via single-spin Metropolis sweeps.
/// SPACE pause  R reset (random spins)  +/- substeps

#include "numerics.hpp"
#include "viz/viz.hpp"
#include "ising/sim.hpp"

// Compile-time alias (zero runtime overhead):
//   double  →  real
using real = double;

static constexpr int N   = 300;
static constexpr int WIN = 700;

int main() {
    IsingLattice<N> lat;
    real T = 2.27;   // near critical temperature (Onsager Tc = 2/ln(1+sqrt(2)) ~ 2.269)
    real h = 0.0;

    num::viz::run("Ising Model", WIN, WIN, [&](num::viz::Frame& f) {
        f.slider("Temperature T", 1.0, 4.0, T);
        f.slider("Field h",      -1.0, 1.0, h);

        if (f.reset_pressed()) {
            lat.random_init();
            lat.set_temperature(T);
            lat.set_field(h);
        }

        f.step([&] {
            lat.set_temperature(T);
            lat.set_field(h);
            lat.sweep();
        });

        f.field(N, [&](int col, int row) -> num::viz::Color {
            return lat.spins[row * N + col] > 0.0
                ? num::viz::Color{240, 240, 240}
                : num::viz::Color{ 20,  20,  20};
        });

        f.textf(WIN - 180, 8, 14, num::viz::kWhite,
                "m = %.3f", lat.magnetization());
    });
}
