/// @file backends/omp/fluid.cpp
/// @brief OpenMP SPH backend  -- parallel for over particles
///
/// Thread safety contract (same principle as src/backends/omp/matrix.cpp):
///   Each parallel section has a unique "owner thread" for writes.
///   Thread i reads particles[j].* fields that were finalised in a prior
///   sequential phase and are not modified during the current section.
///   Thread i writes only to particles[i].* -> no data races without atomics.
///
/// Why per-particle query instead of Newton pairs:
///   iterate_pairs writes to both particles[i] and particles[j].
///   Parallelising over pairs would need atomic float adds  -- expensive and
///   unreadable.  Per-particle query keeps each thread isolated to index i.
///   Thread-level speedup offsets the 2x more pair evaluations for n >~ 500.
///
/// Falls back to seq backend when NUMERICS_HAS_OMP is not defined,
/// identical to the pattern in src/backends/omp/vector.cpp.

#include "impl.hpp"
#include "backends/seq/impl.hpp"
#include "spatial/sph_kernel.hpp"
#include "core/util/integer_pow.hpp"

using namespace num;
using K = num::SPHKernel<2>;
#include <cmath>
#include <algorithm>
#include <cfloat>

#ifdef NUMERICS_HAS_OMP
    #include <omp.h>
#endif

namespace physics::backends::omp {

void compute_density_pressure(std::vector<Particle>& particles,
                              const FluidParams&     params,
                              const SpatialHash&     grid) {
#ifdef NUMERICS_HAS_OMP
    const float h    = params.h;
    const float m    = params.mass;
    const float rho0 = params.rho0;
    const float B    = rho0 * params.c0 * params.c0
                    / static_cast<float>(params.gamma);
    const float supp_sq = 4.0f * h * h;
    const int   n       = static_cast<int>(particles.size());

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        Particle& pi  = particles[i];
        float     rho = 0.0f;
        grid.query(pi.x, pi.y, [&](int j) {
            const float rx = pi.x - particles[j].x;
            const float ry = pi.y - particles[j].y;
            const float r2 = rx * rx + ry * ry;
            if (r2 < supp_sq)
                rho += m * K::W(std::sqrt(r2), h);
        });
        pi.density  = std::max(rho, rho0 * 0.1f);
        pi.pressure = std::max(0.0f,
                               B * (num::ipow<7>(pi.density / rho0) - 1.0f));
    }
#else
    seq::compute_density_pressure(particles, params, grid);
#endif
}

void compute_forces(std::vector<Particle>& particles,
                    const FluidParams&     params,
                    const SpatialHash&     grid) {
#ifdef NUMERICS_HAS_OMP
    const float h       = params.h;
    const float m       = params.mass;
    const float mu      = params.mu;
    const float supp_sq = 4.0f * h * h;
    const float eps2    = 0.01f * h * h;
    const int   n       = static_cast<int>(particles.size());

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        const Particle& pi = particles[i];
        float           ax = params.gx, ay = params.gy;
        grid.query(pi.x, pi.y, [&](int j) {
            if (j == i)
                return;
            const Particle& pj = particles[j];
            const float     rx = pi.x - pj.x, ry = pi.y - pj.y;
            const float     r2 = rx * rx + ry * ry;
            if (r2 >= supp_sq || r2 < 1e-10f)
                return;
            const float r = std::sqrt(r2);

            auto [gx, gy]     = K::Spiky_gradW({rx, ry}, r, h);
            const float pterm = pi.pressure / (pi.density * pi.density)
                                + pj.pressure / (pj.density * pj.density);
            ax -= m * pterm * gx;
            ay -= m * pterm * gy;

            const float lap  = 2.0f * K::Spiky_dW_dr(r, h) * r / (r2 + eps2);
            const float visc = m * mu / (pi.density * pj.density) * lap;
            ax += visc * (pi.evx - pj.evx);
            ay += visc * (pi.evy - pj.evy);
        });
        particles[i].ax = ax;
        particles[i].ay = ay;
    }
#else
    seq::compute_forces(particles, params, grid);
#endif
}

void body_collisions(std::vector<Particle>&        particles,
                     const std::vector<RigidBody>& bodies,
                     const FluidParams&            params) {
#ifdef NUMERICS_HAS_OMP
    const int   n = static_cast<int>(particles.size());
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        Particle& p = particles[i];
        for (const RigidBody& body : bodies) {
            const float dx = p.x - body.x, dy = p.y - body.y;
            const float d = std::sqrt(dx * dx + dy * dy);
            if (d < body.radius && d > 1e-8f) {
                const float nx = dx / d, ny = dy / d;
                p.x            = body.x + body.radius * nx;
                p.y            = body.y + body.radius * ny;
                const float vn = p.vx * nx + p.vy * ny;
                if (vn < 0.0f) {
                    p.vx -= (1.0f + params.restitution) * vn * nx;
                    p.vy -= (1.0f + params.restitution) * vn * ny;
                }
            }
        }
    }
#else
    seq::body_collisions(particles, bodies, params);
#endif
}

void integrate(std::vector<Particle>& particles, const FluidParams& params) {
#ifdef NUMERICS_HAS_OMP
    const float dt = params.dt;
    const int   n  = static_cast<int>(particles.size());
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        Particle& p = particles[i];
        p.vx += p.ax * dt;
        p.vy += p.ay * dt;
        p.x += p.vx * dt;
        p.y += p.vy * dt;
        p.evx = 0.5f * (p.evx + p.vx);
        p.evy = 0.5f * (p.evy + p.vy);
        p.temperature =
            std::clamp(p.temperature + p.dT_dt * dt, -100.0f, 300.0f);
    }
#else
    seq::integrate(particles, params);
#endif
}

void enforce_boundaries(std::vector<Particle>& particles,
                        const FluidParams&     params) {
#ifdef NUMERICS_HAS_OMP
    const float e = params.restitution;
    const int   n = static_cast<int>(particles.size());
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        Particle& p = particles[i];
        if (p.x < params.xmin) {
            p.x  = params.xmin;
            p.vx = std::abs(p.vx) * e;
        }
        if (p.x > params.xmax) {
            p.x  = params.xmax;
            p.vx = -std::abs(p.vx) * e;
        }
        if (p.y < params.ymin) {
            p.y  = params.ymin;
            p.vy = std::abs(p.vy) * e;
        }
        if (p.y > params.ymax) {
            p.y  = params.ymax;
            p.vy = -std::abs(p.vy) * e;
        }
    }
#else
    seq::enforce_boundaries(particles, params);
#endif
}

void update_temp_range(const std::vector<Particle>&  particles,
                       const std::vector<RigidBody>& bodies,
                       float&                        T_min,
                       float&                        T_max) {
#ifdef NUMERICS_HAS_OMP
    if (particles.empty())
        return;
    const int   n  = static_cast<int>(particles.size());
    float       lo = FLT_MAX, hi = -FLT_MAX;
    #pragma omp parallel for reduction(min : lo) reduction(max : hi) \
        schedule(static)
    for (int i = 0; i < n; ++i) {
        lo = std::min(lo, particles[i].temperature);
        hi = std::max(hi, particles[i].temperature);
    }
    T_min = lo;
    T_max = hi;
    for (const RigidBody& b : bodies) { // M bodies  -- sequential, M is small
        T_min = std::min(T_min, b.temperature);
        T_max = std::max(T_max, b.temperature);
    }
#else
    seq::update_temp_range(particles, bodies, T_min, T_max);
#endif
}

} // namespace physics::backends::omp
