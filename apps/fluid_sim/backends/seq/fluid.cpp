/// @file backends/seq/fluid.cpp
/// @brief Sequential SPH backend  -- Newton's 3rd law pair traversal
///
/// All physics loops use grid.iterate_pairs() which visits each unique
/// {i,j} pair once.  Forces and density contributions are accumulated
/// symmetrically to both particles -> O(n*k/2) instead of O(n*k).

#include "impl.hpp"
#include "spatial/sph_kernel.hpp"
#include "core/util/integer_pow.hpp"

using K = num::SPHKernel<2>;
#include <cmath>
#include <algorithm>
#include <cfloat>

using namespace num;

namespace physics::backends::seq {

// Density (Poly6) + pressure (Tait EOS)

void compute_density_pressure(std::vector<Particle>& particles,
                               const FluidParams& params,
                               const SpatialHash& grid) {
    const float h       = params.h;
    const float m       = params.mass;
    const float rho0    = params.rho0;
    const float B       = rho0 * params.c0 * params.c0 / static_cast<float>(params.gamma);
    const float supp_sq = 4.0f * h * h;

    const float W0 = K::W(0.0f, h);
    for (Particle& p : particles)
        p.density = m * W0;

    grid.iterate_pairs([&](int i, int j) {
        const float rx = particles[i].x - particles[j].x;
        const float ry = particles[i].y - particles[j].y;
        const float r2 = rx * rx + ry * ry;
        if (r2 >= supp_sq) return;
        const float w = m * K::W(std::sqrt(r2), h);
        particles[i].density += w;
        particles[j].density += w;
    });

    for (Particle& p : particles) {
        p.density  = std::max(p.density, rho0 * 0.1f);
        p.pressure = std::max(0.0f, B * (num::ipow<7>(p.density / rho0) - 1.0f));
    }
}

// Forces: pressure (Spiky) + viscosity (Morris) + gravity

void compute_forces(std::vector<Particle>& particles,
                    const FluidParams& params,
                    const SpatialHash& grid) {
    const float h       = params.h;
    const float m       = params.mass;
    const float mu      = params.mu;
    const float supp_sq = 4.0f * h * h;
    const float eps2    = 0.01f * h * h;

    for (Particle& p : particles) { p.ax = params.gx; p.ay = params.gy; }

    grid.iterate_pairs([&](int i, int j) {
        Particle& pi = particles[i];
        Particle& pj = particles[j];
        const float rx = pi.x - pj.x, ry = pi.y - pj.y;
        const float r2 = rx * rx + ry * ry;
        if (r2 >= supp_sq || r2 < 1e-10f) return;
        const float r = std::sqrt(r2);

        auto [gx, gy] = K::Spiky_gradW({rx, ry}, r, h);
        const float pterm = pi.pressure / (pi.density * pi.density)
                          + pj.pressure / (pj.density * pj.density);
        const float fpx = -m * pterm * gx,  fpy = -m * pterm * gy;
        pi.ax += fpx;  pi.ay += fpy;
        pj.ax -= fpx;  pj.ay -= fpy;

        const float lap  = 2.0f * K::Spiky_dW_dr(r, h) * r / (r2 + eps2);
        const float visc = m * mu / (pi.density * pj.density) * lap;
        const float fvx  = visc * (pi.evx - pj.evx), fvy = visc * (pi.evy - pj.evy);
        pi.ax += fvx;  pi.ay += fvy;
        pj.ax -= fvx;  pj.ay -= fvy;
    });
}

// Remaining phases  -- all O(n) or O(n*M) with small M

void body_collisions(std::vector<Particle>& particles,
                     const std::vector<RigidBody>& bodies,
                     const FluidParams& params) {
    for (Particle& p : particles) {
        for (const RigidBody& body : bodies) {
            const float dx = p.x - body.x, dy = p.y - body.y;
            const float d  = std::sqrt(dx * dx + dy * dy);
            if (d < body.radius && d > 1e-8f) {
                const float nx = dx / d, ny = dy / d;
                p.x = body.x + body.radius * nx;
                p.y = body.y + body.radius * ny;
                const float vn = p.vx * nx + p.vy * ny;
                if (vn < 0.0f) {
                    p.vx -= (1.0f + params.restitution) * vn * nx;
                    p.vy -= (1.0f + params.restitution) * vn * ny;
                }
            }
        }
    }
}

void integrate(std::vector<Particle>& particles, const FluidParams& params) {
    const float dt = params.dt;
    for (Particle& p : particles) {
        p.vx += p.ax * dt;  p.vy += p.ay * dt;
        p.x  += p.vx * dt;  p.y  += p.vy * dt;
        p.evx = 0.5f * (p.evx + p.vx);
        p.evy = 0.5f * (p.evy + p.vy);
        p.temperature = std::clamp(p.temperature + p.dT_dt * dt, -100.0f, 300.0f);
    }
}

void enforce_boundaries(std::vector<Particle>& particles, const FluidParams& params) {
    const float e = params.restitution;
    for (Particle& p : particles) {
        if (p.x < params.xmin) { p.x = params.xmin; p.vx =  std::abs(p.vx) * e; }
        if (p.x > params.xmax) { p.x = params.xmax; p.vx = -std::abs(p.vx) * e; }
        if (p.y < params.ymin) { p.y = params.ymin; p.vy =  std::abs(p.vy) * e; }
        if (p.y > params.ymax) { p.y = params.ymax; p.vy = -std::abs(p.vy) * e; }
    }
}

void update_temp_range(const std::vector<Particle>& particles,
                       const std::vector<RigidBody>& bodies,
                       float& T_min, float& T_max) {
    if (particles.empty()) return;
    T_min = T_max = particles[0].temperature;
    for (const Particle& p : particles) {
        T_min = std::min(T_min, p.temperature);
        T_max = std::max(T_max, p.temperature);
    }
    for (const RigidBody& b : bodies) {
        T_min = std::min(T_min, b.temperature);
        T_max = std::max(T_max, b.temperature);
    }
}

void integrate_bodies(std::vector<RigidBody>& bodies, const FluidParams& params) {
    const float dt = params.dt, e = params.restitution;
    for (RigidBody& b : bodies) {
        if (b.fixed) continue;
        b.vx += params.gx * dt;  b.vy += params.gy * dt;
        b.x  += b.vx * dt;       b.y  += b.vy * dt;
        const float r = b.radius;
        if (b.x - r < params.xmin) { b.x = params.xmin + r; b.vx =  std::abs(b.vx) * e; }
        if (b.x + r > params.xmax) { b.x = params.xmax - r; b.vx = -std::abs(b.vx) * e; }
        if (b.y - r < params.ymin) { b.y = params.ymin + r; b.vy =  std::abs(b.vy) * e; }
        if (b.y + r > params.ymax) { b.y = params.ymax - r; b.vy = -std::abs(b.vy) * e; }
    }
}

} // namespace physics::backends::seq
