/// @file apps/ns_demo/main.cpp
/// @brief 2D incompressible Navier-Stokes -- Kelvin-Helmholtz shear layer.
///
/// Chorin projection on a MAC grid.  Vorticity shown as a diverging colormap;
/// 3000 passive tracers reveal the flow structure.
/// SPACE pause  R reset  +/- substeps

#include "numerics.hpp"
#include "viz/viz.hpp"
#include "ns_demo/sim.hpp"
#include <vector>
#include <cstdlib>

// Compile-time alias (zero runtime overhead):
//   static_cast<float>(x)  →  cast<float>(x)
template<class To, class From>
constexpr To cast(From x) {
    return static_cast<To>(x);
}

static constexpr ns::idx  N        = 256;
static constexpr ns::real DT       = 0.5 / cast<ns::real>(N);
static constexpr ns::real NU       = 0.0;
static constexpr int      NPART    = 3000;
static constexpr int      MAX_AGE  = 300;
static constexpr int      WIN      = 900;
static constexpr float    OMEGA_SC = 20.0f; // vorticity scale for colormap

struct Tracer {
    float x = 0, y = 0;
    int   age = 0;
};

static void respawn(Tracer& p) {
    p.x   = cast<float>(rand()) / RAND_MAX;
    p.y   = cast<float>(rand()) / RAND_MAX;
    p.age = 0;
}

int main() {
    ns::NSSolver solver(N, DT, NU);
    solver.init_shear_layer();

    std::vector<Tracer> tracers(NPART);
    for (auto& p : tracers)
        respawn(p);

    num::viz::run("NS: Kelvin-Helmholtz Shear Layer",
                  WIN,
                  WIN,
                  [&](num::viz::Frame& f) {
                      if (f.reset_pressed()) {
                          solver.init_shear_layer();
                          for (auto& p : tracers)
                              respawn(p);
                      }

                      f.step([&] {
                          solver.step();

                          float frame_dt = cast<float>(DT);
                          for (auto& p : tracers) {
                              float u = cast<float>(solver.interp_u(p.x, p.y));
                              float v = cast<float>(solver.interp_v(p.x, p.y));
                              p.x     = fmodf(p.x + u * frame_dt + 1.0f, 1.0f);
                              p.y     = fmodf(p.y + v * frame_dt + 1.0f, 1.0f);
                              if (++p.age > MAX_AGE)
                                  respawn(p);
                          }
                      });

                      // Vorticity field
                      f.field(cast<int>(N), [&](int col, int row) {
                          float omega = cast<float>(solver.vorticity(col, row));
                          return num::viz::diverging_color(omega / OMEGA_SC);
                      });

                      // Passive tracers
                      for (const auto& p : tracers) {
                          float age_t = 1.0f - cast<float>(p.age) / MAX_AGE;
                          f.dot(p.x * WIN,
                                p.y * WIN,
                                {255, 255, 255, cast<uint8_t>(age_t * 160)},
                                1.2f);
                      }
                  });
}
