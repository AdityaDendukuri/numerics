/// @file apps/fluid_sim/main.cpp
/// @brief 2D weakly-compressible SPH fluid -- dam break.
///
/// Left half of domain filled with water; one hot and one cold rigid body
/// on the right.  Heat diffuses between particles and bodies.
/// SPACE pause  R reset  +/- substeps

#include "numerics.hpp"
#include "viz/viz.hpp"
#include "fluid_sim/sim.hpp"
#include <algorithm>
#include <cmath>

// Compile-time alias
//   static_cast<float>(x)  →  cast<float>(x)
//   static_cast<int>(x)    →  cast<int>(x)
template<class To, class From>
constexpr To cast(From x) {
    return static_cast<To>(x);
}

static constexpr int   WIN   = 900;
static constexpr float DOM_W = 1.0f;
static constexpr float DOM_H = 0.7f;
static constexpr float SCALE = WIN / DOM_W; // px/m
static constexpr float PRAD  = 4.0f;        // particle draw radius (px)

static void sim_to_screen(float sx, float sy, float& px, float& py) {
    px = sx * SCALE;
    py = (DOM_H - sy) * SCALE;
}

static physics::FluidParams make_params() {
    physics::FluidParams p;
    p.h           = 0.025f;
    p.rho0        = 1000.0f;
    p.gamma       = 7;
    p.c0          = 10.0f;
    p.mu          = 8.0f;
    p.mass        = p.rho0 * (0.8f * p.h) * (0.8f * p.h);
    p.gx          = 0.0f;
    p.gy          = -9.81f;
    p.dt          = 0.001f;
    p.xmin        = 0.0f;
    p.xmax        = DOM_W;
    p.ymin        = 0.0f;
    p.ymax        = DOM_H;
    p.restitution = 0.01f;
    p.alpha_T     = 0.005f;
    p.h_conv      = 8.0f;
    return p;
}

static void setup(physics::FluidSolver& solver) {
    solver.clear();
    const float h  = solver.params().h;
    const float dx = 0.8f * h;

    for (float x = 0.03f; x < 0.50f; x += dx)
        for (float y = 0.03f; y < 0.65f; y += dx) {
            float jx = 0.001f * ((cast<int>(x * 1000) % 7) - 3);
            float jy = 0.001f * ((cast<int>(y * 1000) % 5) - 2);
            solver.add_particle(x + jx, y + jy, 0.0f, 0.0f, 20.0f);
        }

    physics::RigidBody hot{};
    hot.x              = 0.74f;
    hot.y              = 0.22f;
    hot.radius         = 0.10f;
    hot.temperature    = 90.0f;
    hot.mass           = 5000.0f;
    hot.fixed          = true;
    hot.is_heat_source = true;
    solver.add_body(hot);

    physics::RigidBody cold{};
    cold.x              = 0.78f;
    cold.y              = 0.57f;
    cold.radius         = 0.08f;
    cold.temperature    = 0.0f;
    cold.mass           = 5000.0f;
    cold.fixed          = true;
    cold.is_heat_source = true;
    solver.add_body(cold);
}

int main() {
    physics::FluidSolver solver(make_params());
    setup(solver);

    num::viz::run(
        "2D SPH Fluid: Dam Break",
        WIN,
        cast<int>(WIN * DOM_H / DOM_W),
        [&](num::viz::Frame& f) {
            if (f.reset_pressed())
                setup(solver);

            f.step([&] { solver.step(); });

            float T_min = solver.min_temp();
            float T_max = solver.max_temp();
            float T_rng = T_max - T_min + 1e-4f;

            // Domain boundary
            f.rect_outline(0,
                           0,
                           cast<float>(f.width),
                           cast<float>(f.height),
                           {60, 60, 80, 255});

            // Rigid bodies
            for (const auto& body : solver.bodies()) {
                float px, py;
                sim_to_screen(body.x, body.y, px, py);
                float r  = body.radius * SCALE;
                auto  fc = num::viz::heat_color((body.temperature - T_min)
                                               / T_rng);
                fc.a     = 180;
                f.dot(px, py, fc, r);
                f.circle(px, py, r, {220, 220, 220, 200});
                const char* label = body.temperature > 50.0f ? "HOT" : "COLD";
                f.text(label,
                       px - 14,
                       py - 8,
                       14,
                       body.temperature > 50.0f ? num::viz::Color{255, 80, 80}
                                                : num::viz::kSkyBlue);
            }

            // Particles
            for (const auto& p : solver.particles()) {
                float px, py;
                sim_to_screen(p.x, p.y, px, py);
                f.dot(px,
                      py,
                      num::viz::heat_color((p.temperature - T_min) / T_rng),
                      PRAD);
            }

            f.textf(8,
                    8,
                    18,
                    num::viz::kWhite,
                    "particles: %d",
                    cast<int>(solver.particles().size()));
        },
        {18, 18, 28, 255});
}
