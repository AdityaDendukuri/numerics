/// @file apps/nbody/main.cpp
/// @brief Gravitational N-body simulation -- galaxy collapse.
///
/// N=200 bodies, symplectic Verlet integrator, trail rendering.
/// SPACE pause  R reset  +/- substeps

#include "numerics.hpp"
#include "viz/viz.hpp"
#include "nbody/sim.hpp"
#include <deque>
#include <vector>

// Compile-time aliases (zero runtime overhead; verbose C++ shown on the right):
//   static_cast<float>(x)  →  cast<float>(x)
//   static_cast<int>(x)    →  cast<int>(x)
//   std::move(x)           →  move(x)
using real = double;
template<class To, class From> constexpr To   cast(From x)  { return static_cast<To>(x); }
template<class T>              constexpr auto  move(T&& x)  { return std::move(x); }

static constexpr int   WIN   = 900;
static constexpr real  DT    = 0.001;
static constexpr int   TRAIL = 40;
static constexpr float SCALE = 130.0f;  // pixels per simulation unit

struct V2 { float x, y; };

static V2 to_screen(real sx, real sy) {
    return { WIN * 0.5f + cast<float>(sx) * SCALE,
             WIN * 0.5f - cast<float>(sy) * SCALE };
}

int main() {
    nbody::NBodySim sim;
    sim.reset(nbody::Scenario::Galaxy);

    std::vector<std::deque<V2>> trails(sim.n());

    num::viz::run("N-body: Galaxy Collapse", WIN, WIN, [&](num::viz::Frame& f) {
        if (f.reset_pressed()) {
            sim.reset(nbody::Scenario::Galaxy);
            trails.assign(sim.n(), {});
        }

        f.step([&] {
            if (sim.enable_merges) {
                auto ops = sim.check_merges();
                for (auto& [j, last_] : ops) {
                    if (j != last_) std::swap(trails[j], trails[last_]);
                    trails.pop_back();
                }
            }
            sim.step(DT);

            num::Vec2ConstView pos{sim.q};
            for (int i = 0; i < sim.n(); ++i) {
                auto p = to_screen(pos.x(i), pos.y(i));
                auto& tr = trails[i];
                tr.push_front(p);
                if (cast<int>(tr.size()) > TRAIL) tr.pop_back();
            }
        });

        // Trails
        for (int i = 0; i < sim.n(); ++i) {
            auto& tr  = trails[i];
            auto  col = num::viz::unpack(sim.bodies[i].color);
            for (int k = 0; k + 1 < cast<int>(tr.size()); ++k) {
                float alpha = 1.0f - cast<float>(k) / TRAIL;
                col.a = cast<uint8_t>(alpha * alpha * 180);
                f.line(tr[k].x, tr[k].y, tr[k+1].x, tr[k+1].y, col);
            }
        }

        // Bodies (three concentric circles for glow)
        num::Vec2ConstView pos{sim.q};
        for (int i = 0; i < sim.n(); ++i) {
            auto  sc  = to_screen(pos.x(i), pos.y(i));
            auto  col = num::viz::unpack(sim.bodies[i].color);
            float r   = sim.bodies[i].display_radius;
            f.dot(sc.x, sc.y, {col.r, col.g, col.b, 40},  r * 2.2f);
            f.dot(sc.x, sc.y, {col.r, col.g, col.b, 100}, r * 1.4f);
            f.dot(sc.x, sc.y, col, r);
        }

        f.textf(8, 8, 16, num::viz::kWhite, "bodies: %d  substeps: %d",
                sim.n(), f.substeps);
    });
}
