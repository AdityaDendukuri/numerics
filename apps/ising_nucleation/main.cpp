/// @file apps/ising_nucleation/main.cpp
/// @brief 2D Ising model -- umbrella-sampled nucleation with live sliders.
///
/// Umbrella sampling constrains the largest spin-down cluster (nucleus) to
/// a target size window [target - half, target + half].
/// Reproduces Brendel et al. (2005) nucleation experiment.
///
/// Sliders: temperature T, external field h, nucleus target size, window
/// half-width. Spin up = light gray, nucleus cluster = red, spin down = dark.
/// SPACE pause  R reset  +/- substeps

#include "numerics.hpp"
#include "viz/viz.hpp"
#include "ising/sim.hpp"
#include <vector>

// Compile-time alias (zero runtime overhead):
//   double  →  real
using real = double;

static constexpr int N   = 300;
static constexpr int WIN = 700;

int main() {
    IsingLattice<N> lat;
    real            T      = 0.58; // well below Tc  (metastable droplet regime)
    real            h      = 0.10; // small external field driving nucleation
    real            target = 15.0; // target nucleus size (spins)
    real            half   = 15.0; // window half-width

    num::ClusterResult det;
    det.id.assign(N * N, -2);

    auto do_reset = [&] {
        lat.set_temperature(T);
        lat.set_field(h);
        lat.generate_nucleus(target);
        det.id.assign(N * N, -2);
    };
    do_reset();

    num::viz::run("Ising: Nucleation (Umbrella Sampling)",
                  WIN,
                  WIN,
                  [&](num::viz::Frame& f) {
                      f.slider("Temperature T", 0.3, 0.9, T);
                      f.slider("Field h", 0.01, 0.5, h);
                      f.slider("Nucleus target", 5.0, 60.0, target);
                      f.slider("Window half-width", 5.0, 30.0, half);

                      if (f.reset_pressed())
                          do_reset();

                      f.step([&] {
                          lat.set_temperature(T);
                          lat.set_field(h);
                          int lo = static_cast<int>(target - half);
                          int hi = static_cast<int>(target + half);
                          lat.sweep_umbrella(lo < 1 ? 1 : lo, hi, det);
                      });

                      f.field(N, [&](int col, int row) -> num::viz::Color {
                          int idx = row * N + col;
                          if (lat.spins[idx] > 0.0)
                              return {240, 240, 240, 255};
                          if (det.id[idx] == det.largest_id)
                              return {220, 50, 50, 255};
                          return {15, 15, 15, 255};
                      });

                      // Nucleus size overlay
                      f.textf(WIN - 190,
                              8,
                              14,
                              num::viz::kWhite,
                              "nucleus: %d spins",
                              det.largest_size);
                  });
}
