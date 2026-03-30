/// @file apps/fluid_sim_3d/main.cpp
/// @brief 3D weakly-compressible SPH -- dual opposing hoses.
///
/// One cold hose and one hot hose, both always active, narrow cone (~2 deg).
/// Particles colored by temperature (blue=cold, red=hot).
/// SPACE pause  R reset  +/- substeps

#include "numerics.hpp"
#include "viz/viz.hpp"
#include "fluid_sim_3d/sim.hpp"
#include <cmath>

// Compile-time alias (zero runtime overhead):
//   static_cast<float>(x)  →  cast<float>(x)
//   static_cast<int>(x)    →  cast<int>(x)
template<class To, class From> constexpr To cast(From x) { return static_cast<To>(x); }

static constexpr float kPI = 3.14159265f;
static constexpr int   WIN = 900;

static physics::FluidParams3D make_params() {
    physics::FluidParams3D p;
    p.h           = 0.05f;
    p.rho0        = 1000.0f;
    p.gamma       = 7;
    p.c0          = 10.0f;
    p.mu          = 10.0f;
    p.mass        = p.rho0 * (0.8f*p.h) * (0.8f*p.h) * (0.8f*p.h);
    p.gx = 0.0f;  p.gy = -9.81f;  p.gz = 0.0f;
    p.dt          = 0.001f;
    p.xmin = p.ymin = p.zmin = 0.0f;
    p.xmax = p.ymax = p.zmax = 0.8f;
    p.restitution = 0.01f;
    p.alpha_T     = 0.005f;
    p.h_conv      = 8.0f;
    return p;
}

// Hose emitter
struct Hose {
    Vector3 source, dir, perp_u, perp_v;
    float temperature, speed, cone_half;

    static Vector3 norm(Vector3 v) {
        float len = sqrtf(v.x*v.x + v.y*v.y + v.z*v.z);
        return len < 1e-8f ? Vector3{0,1,0} : Vector3{v.x/len, v.y/len, v.z/len};
    }
    static Vector3 cross(Vector3 a, Vector3 b) {
        return {a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x};
    }

    void init(Vector3 src, Vector3 target, float temp, float spd, float half_angle) {
        source = src; temperature = temp; speed = spd; cone_half = half_angle;
        dir    = norm({target.x-src.x, target.y-src.y, target.z-src.z});
        Vector3 ref = fabsf(dir.y) < 0.9f ? Vector3{0,1,0} : Vector3{1,0,0};
        perp_u = norm(cross(dir, ref));
        perp_v = cross(dir, perp_u);
    }

    void emit(physics::FluidSolver3D& solver, int count, int max_n = 1800) const {
        if (cast<int>(solver.particles().size()) >= max_n) return;
        const float max_r = tanf(cone_half);
        for (int i = 0; i < count; ++i) {
            float angle = cast<float>(GetRandomValue(0, 10000)) / 10000.0f * 2.0f * kPI;
            float r     = cast<float>(GetRandomValue(0, 10000)) / 10000.0f * max_r;
            float su = cosf(angle) * r, sv = sinf(angle) * r;
            Vector3 d = norm({dir.x + su*perp_u.x + sv*perp_v.x,
                              dir.y + su*perp_u.y + sv*perp_v.y,
                              dir.z + su*perp_u.z + sv*perp_v.z});
            float jx = cast<float>(GetRandomValue(-100,100)) * 0.00008f;
            float jy = cast<float>(GetRandomValue(-100,100)) * 0.00008f;
            float jz = cast<float>(GetRandomValue(-100,100)) * 0.00008f;
            solver.add_particle(source.x+jx, source.y+jy, source.z+jz,
                                d.x*speed, d.y*speed, d.z*speed, temperature);
        }
    }
};

int main() {
    physics::FluidSolver3D solver(make_params());

    Hose cold, hot;
    cold.init({0.77f, 0.76f, 0.02f}, {0.4f, 0.4f, 0.4f},  5.0f, 1.8f, kPI / 90.0f);
    hot .init({0.03f, 0.76f, 0.02f}, {0.4f, 0.4f, 0.4f}, 80.0f, 1.8f, kPI / 90.0f);

    num::viz::run("3D SPH Fluid: Opposing Hoses", WIN, WIN,
        [&](num::viz::Frame& f) {
            if (f.reset_pressed()) {
                solver = physics::FluidSolver3D(make_params());
            }

            f.step([&] {
                cold.emit(solver, 2);
                hot .emit(solver, 2);
                solver.step();
            });

            float T_min = solver.min_temp();
            float T_max = solver.max_temp();

            f.begin3d({1.5f, 1.2f, 1.8f,   0.4f, 0.4f, 0.4f});

            DrawCubeWires({0.4f, 0.4f, 0.4f}, 0.8f, 0.8f, 0.8f, {60, 60, 80, 200});

            // Nozzles
            f.sphere3d(cold.source.x, cold.source.y, cold.source.z,
                       0.022f, num::viz::kSkyBlue);
            f.sphere3d(hot.source.x, hot.source.y, hot.source.z,
                       0.022f, {255, 140, 20, 255});

            // Particles
            float T_rng = T_max - T_min + 1e-4f;
            for (const auto& p : solver.particles()) {
                auto c = num::viz::heat_color((p.temperature - T_min) / T_rng);
                f.sphere3d(p.x, p.y, p.z, 0.016f, c);
            }

            f.end3d();

            f.textf(8, 8, 18, num::viz::kWhite,
                    "particles: %d", cast<int>(solver.particles().size()));
        },
        {18, 18, 28, 255});
}
