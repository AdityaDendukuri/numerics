/// @file backends/seq/heat.cpp
/// @brief Sequential heat diffusion backend (3D)

#include "impl.hpp"
#include "spatial/sph_kernel.hpp"

using K = num::SPHKernel<3>;
#include <cmath>
#include <algorithm>

namespace physics::backends::seq {

void heat_compute(std::vector<Particle3D>&        particles,
                  const std::vector<RigidBody3D>& bodies,
                  const SpatialHash3D&            grid,
                  const HeatParams3D&             params) {
    const float h       = params.h;
    const float alpha   = params.alpha_T;
    const float m       = params.mass;
    const float h_conv  = params.h_conv;
    const float supp_sq = 4.0f * h * h;
    const float eps2    = 0.01f * h * h;
    const int   n       = static_cast<int>(particles.size());

    for (int i = 0; i < n; ++i) {
        Particle3D& pi = particles[i];
        float       dT = 0.0f;

        grid.query(pi.x, pi.y, pi.z, [&](int j) {
            if (j == i)
                return;
            const Particle3D& pj = particles[j];
            const float rx = pi.x - pj.x, ry = pi.y - pj.y, rz = pi.z - pj.z;
            const float r2 = rx * rx + ry * ry + rz * rz;
            if (r2 >= supp_sq || r2 < 1e-10f)
                return;
            const float r             = std::sqrt(r2);
            const float rij_dot_gradW = K::dW_dr(r, h) * r;
            dT += 2.0f * alpha * (m / pj.density)
                  * (pi.temperature - pj.temperature) * rij_dot_gradW
                  / (r2 + eps2);
        });

        for (const RigidBody3D& body : bodies) {
            const float dx = pi.x - body.x, dy = pi.y - body.y,
                        dz = pi.z - body.z;
            const float d  = std::sqrt(dx * dx + dy * dy + dz * dz);
            if (d < body.radius + 2.0f * h) {
                const float excess = std::max(0.0f, d - body.radius);
                const float phi    = 1.0f - excess / (2.0f * h);
                dT += h_conv * (body.temperature - pi.temperature) * phi;
            }
        }

        pi.dT_dt = dT;
    }
}

} // namespace physics::backends::seq
