/// @file backends/seq/fluid.cpp
/// @brief Sequential 3D SPH backend  -- Newton's 3rd law pair traversal

#include "impl.hpp"
#include "spatial/sph_kernel.hpp"

using K = num::SPHKernel<3>;
#include "core/util/integer_pow.hpp"
#include <cmath>
#include <algorithm>
#include <cfloat>

namespace physics::backends::seq {

void compute_density_pressure(std::vector<Particle3D>& particles,
                              const FluidParams3D&     params,
                              const SpatialHash3D&     grid) {
    const float h    = params.h;
    const float m    = params.mass;
    const float rho0 = params.rho0;
    const float B    = rho0 * params.c0 * params.c0
                    / static_cast<float>(params.gamma);
    const float supp_sq = 4.0f * h * h;

    const float W0 = K::W(0.0f, h);
    for (Particle3D& p : particles)
        p.density = m * W0;

    grid.iterate_pairs([&](int i, int j) {
        const float rx = particles[i].x - particles[j].x;
        const float ry = particles[i].y - particles[j].y;
        const float rz = particles[i].z - particles[j].z;
        const float r2 = rx * rx + ry * ry + rz * rz;
        if (r2 >= supp_sq)
            return;
        const float w = m * K::W(std::sqrt(r2), h);
        particles[i].density += w;
        particles[j].density += w;
    });

    for (Particle3D& p : particles) {
        p.density  = std::max(p.density, rho0 * 0.1f);
        p.pressure = std::max(0.0f,
                              B * (num::ipow<7>(p.density / rho0) - 1.0f));
    }
}

void compute_forces(std::vector<Particle3D>& particles,
                    const FluidParams3D&     params,
                    const SpatialHash3D&     grid) {
    const float h       = params.h;
    const float m       = params.mass;
    const float mu      = params.mu;
    const float supp_sq = 4.0f * h * h;
    const float eps2    = 0.01f * h * h;

    for (Particle3D& p : particles) {
        p.ax = params.gx;
        p.ay = params.gy;
        p.az = params.gz;
    }

    grid.iterate_pairs([&](int i, int j) {
        Particle3D& pi = particles[i];
        Particle3D& pj = particles[j];
        const float rx = pi.x - pj.x, ry = pi.y - pj.y, rz = pi.z - pj.z;
        const float r2 = rx * rx + ry * ry + rz * rz;
        if (r2 >= supp_sq || r2 < 1e-10f)
            return;
        const float r = std::sqrt(r2);

        auto [gx, gy, gz] = K::Spiky_gradW({rx, ry, rz}, r, h);
        const float pterm = pi.pressure / (pi.density * pi.density)
                            + pj.pressure / (pj.density * pj.density);
        const float fpx = -m * pterm * gx, fpy = -m * pterm * gy,
                    fpz = -m * pterm * gz;
        pi.ax += fpx;
        pi.ay += fpy;
        pi.az += fpz;
        pj.ax -= fpx;
        pj.ay -= fpy;
        pj.az -= fpz;

        const float lap  = 2.0f * K::Spiky_dW_dr(r, h) * r / (r2 + eps2);
        const float visc = m * mu / (pi.density * pj.density) * lap;
        const float fvx  = visc * (pi.evx - pj.evx);
        const float fvy  = visc * (pi.evy - pj.evy);
        const float fvz  = visc * (pi.evz - pj.evz);
        pi.ax += fvx;
        pi.ay += fvy;
        pi.az += fvz;
        pj.ax -= fvx;
        pj.ay -= fvy;
        pj.az -= fvz;
    });
}

void body_collisions(std::vector<Particle3D>&        particles,
                     const std::vector<RigidBody3D>& bodies,
                     const FluidParams3D&            params) {
    for (Particle3D& p : particles) {
        for (const RigidBody3D& body : bodies) {
            const float dx = p.x - body.x, dy = p.y - body.y, dz = p.z - body.z;
            const float d = std::sqrt(dx * dx + dy * dy + dz * dz);
            if (d < body.radius && d > 1e-8f) {
                const float nx = dx / d, ny = dy / d, nz = dz / d;
                p.x            = body.x + body.radius * nx;
                p.y            = body.y + body.radius * ny;
                p.z            = body.z + body.radius * nz;
                const float vn = p.vx * nx + p.vy * ny + p.vz * nz;
                if (vn < 0.0f) {
                    const float c = (1.0f + params.restitution) * vn;
                    p.vx -= c * nx;
                    p.vy -= c * ny;
                    p.vz -= c * nz;
                }
            }
        }
    }
}

void integrate(std::vector<Particle3D>& particles,
               const FluidParams3D&     params) {
    const float dt = params.dt;
    for (Particle3D& p : particles) {
        p.vx += p.ax * dt;
        p.vy += p.ay * dt;
        p.vz += p.az * dt;
        p.x += p.vx * dt;
        p.y += p.vy * dt;
        p.z += p.vz * dt;
        p.evx = 0.5f * (p.evx + p.vx);
        p.evy = 0.5f * (p.evy + p.vy);
        p.evz = 0.5f * (p.evz + p.vz);
        p.temperature =
            std::clamp(p.temperature + p.dT_dt * dt, -100.0f, 300.0f);
    }
}

void enforce_boundaries(std::vector<Particle3D>& particles,
                        const FluidParams3D&     params) {
    const float e = params.restitution;
    for (Particle3D& p : particles) {
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
        if (p.z < params.zmin) {
            p.z  = params.zmin;
            p.vz = std::abs(p.vz) * e;
        }
        if (p.z > params.zmax) {
            p.z  = params.zmax;
            p.vz = -std::abs(p.vz) * e;
        }
    }
}

void update_temp_range(const std::vector<Particle3D>&  particles,
                       const std::vector<RigidBody3D>& bodies,
                       float&                          T_min,
                       float&                          T_max) {
    if (particles.empty())
        return;
    T_min = T_max = particles[0].temperature;
    for (const Particle3D& p : particles) {
        T_min = std::min(T_min, p.temperature);
        T_max = std::max(T_max, p.temperature);
    }
    for (const RigidBody3D& b : bodies) {
        T_min = std::min(T_min, b.temperature);
        T_max = std::max(T_max, b.temperature);
    }
}

void integrate_bodies(std::vector<RigidBody3D>& bodies,
                      const FluidParams3D&      params) {
    const float dt = params.dt, e = params.restitution;
    for (RigidBody3D& b : bodies) {
        if (b.fixed)
            continue;
        b.vx += params.gx * dt;
        b.vy += params.gy * dt;
        b.vz += params.gz * dt;
        b.x += b.vx * dt;
        b.y += b.vy * dt;
        b.z += b.vz * dt;
        const float r = b.radius;
        if (b.x - r < params.xmin) {
            b.x  = params.xmin + r;
            b.vx = std::abs(b.vx) * e;
        }
        if (b.x + r > params.xmax) {
            b.x  = params.xmax - r;
            b.vx = -std::abs(b.vx) * e;
        }
        if (b.y - r < params.ymin) {
            b.y  = params.ymin + r;
            b.vy = std::abs(b.vy) * e;
        }
        if (b.y + r > params.ymax) {
            b.y  = params.ymax - r;
            b.vy = -std::abs(b.vy) * e;
        }
        if (b.z - r < params.zmin) {
            b.z  = params.zmin + r;
            b.vz = std::abs(b.vz) * e;
        }
        if (b.z + r > params.zmax) {
            b.z  = params.zmax - r;
            b.vz = -std::abs(b.vz) * e;
        }
    }
}

} // namespace physics::backends::seq
