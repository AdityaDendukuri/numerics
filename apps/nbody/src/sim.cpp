/// @file src/sim.cpp
/// @brief NBodySim implementation -- physics, integrators, scenarios.

#include "nbody/sim.hpp"
#include <algorithm>
#include <random>

namespace nbody {

// Color mapped from body mass: blue-white (low) to yellow (mid) to orange-red
// (high)
uint32_t mass_color(double m) {
    const double t = std::min(m / 5.0, 1.0);
    uint8_t      r_, g_, b_;
    if (t < 0.5) {
        float u = float(t * 2.0);
        r_      = uint8_t(120 + u * 135); // 120..255
        g_      = uint8_t(200 + u * 38);  // 200..238
        b_      = uint8_t(255 - u * 85);  // 255..170
    } else {
        float u = float((t - 0.5) * 2.0);
        r_      = 255;
        g_      = uint8_t(238 - u * 136); // 238..102
        b_      = uint8_t(170 - u * 136); // 170..34
    }
    return (uint32_t(r_) << 24) | (uint32_t(g_) << 16) | (uint32_t(b_) << 8)
           | 0xFFu;
}

static constexpr float DISP_SCALE = 6.5f;  // pixels for mass = 1.0
static constexpr float PHYS_SCALE = 0.05f; // phys units for mass = 1.0

float disp_r(double m) {
    return DISP_SCALE * std::cbrt(float(m));
}
float phys_r(double m) {
    return PHYS_SCALE * std::cbrt(float(m));
}

void NBodySim::make_accel(const num::Vector& pos, num::Vector& acc) const {
    const int nb = n();
    std::fill(acc.begin(), acc.end(), 0.0);
    num::Vec2ConstView p{pos};
    num::Vec2View      a{acc};
    for (int i = 0; i < nb; ++i) {
        for (int j = 0; j < nb; ++j) {
            if (i == j)
                continue;
            double dx  = p.x(j) - p.x(i);
            double dy  = p.y(j) - p.y(i);
            double r2  = dx * dx + dy * dy + soft * soft;
            double fac = G * bodies[j].mass / (r2 * std::sqrt(r2));
            a.x(i) += fac * dx;
            a.y(i) += fac * dy;
        }
    }
}

double NBodySim::kinetic_energy() const {
    num::Vec2ConstView vel{v};
    double             KE = 0.0;
    for (int i = 0; i < n(); ++i)
        KE += 0.5 * bodies[i].mass
              * (vel.x(i) * vel.x(i) + vel.y(i) * vel.y(i));
    return KE;
}

double NBodySim::potential_energy() const {
    num::Vec2ConstView pos{q};
    double             PE = 0.0;
    for (int i = 0; i < n(); ++i) {
        for (int j = i + 1; j < n(); ++j) {
            double dx = pos.x(j) - pos.x(i);
            double dy = pos.y(j) - pos.y(i);
            PE -= G * bodies[i].mass * bodies[j].mass
                  / std::sqrt(dx * dx + dy * dy + soft * soft);
        }
    }
    return PE;
}

void NBodySim::step(double dt) {
    auto accel = [this](const num::Vector& pos, num::Vector& acc) {
        make_accel(pos, acc);
    };
    num::ODEParams        p = {.t0 = t, .tf = t + dt, .h = dt};
    num::SymplecticResult res;
    switch (integrator) {
        case Integrator::Verlet:
            res = num::verlet(accel, q, v, p).run();
            break;
        case Integrator::Yoshida4:
            res = num::yoshida4(accel, q, v, p).run();
            break;
        case Integrator::RK4:
            res = num::rk4_2nd(accel, q, v, p).run();
            break;
    }
    q = std::move(res.q);
    v = std::move(res.v);
    t += dt;
}

std::vector<std::pair<int, int>> NBodySim::check_merges() {
    std::vector<std::pair<int, int>> ops;
    int                              i = 0;
    while (i < n() && n() > 1) {
        bool          merged = false;
        num::Vec2View pos{q}, vel{v};
        for (int j = i + 1; j < n(); ++j) {
            double dx  = pos.x(i) - pos.x(j);
            double dy  = pos.y(i) - pos.y(j);
            double mrd = bodies[i].phys_radius + bodies[j].phys_radius;
            if (dx * dx + dy * dy >= mrd * mrd)
                continue;

            // Merge j into i -- conserve mass and momentum
            double mi = bodies[i].mass, mj = bodies[j].mass, mt = mi + mj;
            pos.x(i)                 = (mi * pos.x(i) + mj * pos.x(j)) / mt;
            pos.y(i)                 = (mi * pos.y(i) + mj * pos.y(j)) / mt;
            vel.x(i)                 = (mi * vel.x(i) + mj * vel.x(j)) / mt;
            vel.y(i)                 = (mi * vel.y(i) + mj * vel.y(j)) / mt;
            bodies[i].mass           = mt;
            bodies[i].display_radius = disp_r(mt);
            bodies[i].phys_radius    = phys_r(mt);
            bodies[i].color          = mass_color(mt);

            // Swap j with the last body and shrink arrays
            int last = n() - 1;
            ops.push_back({j, last});
            if (j != last) {
                std::swap(bodies[j], bodies[last]);
                pos.x(j) = pos.x(last);
                pos.y(j) = pos.y(last);
                vel.x(j) = vel.x(last);
                vel.y(j) = vel.y(last);
            }
            bodies.pop_back();

            // Rebuild q and v at the new (smaller) size
            int         nn = n();
            num::Vector nq(2 * nn, 0.0), nv(2 * nn, 0.0);
            for (int k = 0; k < 2 * nn; ++k) {
                nq[k] = q[k];
                nv[k] = v[k];
            }
            q = std::move(nq);
            v = std::move(nv);

            merged = true;
            break; // restart j-scan for body i (it may now touch another body)
        }
        if (!merged)
            ++i;
    }
    return ops;
}

void NBodySim::reset(Scenario s) {
    t             = 0.0;
    G             = 1.0;
    enable_merges = false;

    if (s == Scenario::Figure8) {
        soft   = 1e-4;
        bodies = {
            {1.0, 10.0f, 0xFF5555FF},
            {1.0, 10.0f, 0x55FF55FF},
            {1.0, 10.0f, 0x5599FFFF},
        };
        q = {-0.97000436, 0.24308753, 0.97000436, -0.24308753, 0.0, 0.0};
        v = {0.46620368,
             0.43236573,
             0.46620368,
             0.43236573,
             -0.93240737,
             -0.86473146};

    } else if (s == Scenario::SolarSystem) {
        soft              = 1e-3;
        const double Msun = 1000.0;
        bodies            = {
            {Msun, 18.0f, 0xFFDD33FF},
            {1.0, 4.0f, 0xAAAAAAFF},
            {1.0, 6.0f, 0x44AAFFFF},
            {1.0, 5.0f, 0xFF6644FF},
            {10.0, 11.0f, 0xFFAA44FF},
        };
        const double radii[] = {0.0, 0.22, 0.40, 0.60, 1.10};
        const int    nb      = 5;
        q                    = num::Vector(2 * nb, 0.0);
        v                    = num::Vector(2 * nb, 0.0);
        num::Vec2View pos{q}, vel{v};
        for (int i = 1; i < nb; ++i) {
            pos.x(i) = radii[i];
            vel.y(i) = std::sqrt(G * Msun / radii[i]);
        }

    } else if (s == Scenario::BinaryPlus) {
        soft               = 1e-3;
        const double Mstar = 50.0;
        bodies             = {
            {Mstar, 13.0f, 0xFF7744FF},
            {Mstar, 13.0f, 0x44AAFFFF},
            {0.01, 5.0f, 0xAAFF88FF},
        };
        const double sep   = 0.3;
        const double omega = std::sqrt(G * 2.0 * Mstar / (sep * sep * sep));
        double       vs    = omega * sep * 0.5;
        double       r_tp  = 1.5;
        double       v_tp  = std::sqrt(G * 2.0 * Mstar / r_tp) * 0.85;
        q                  = {-sep * 0.5, 0.0, sep * 0.5, 0.0, r_tp, 0.0};
        v                  = {0.0, -vs, 0.0, vs, 0.0, v_tp};

    } else { // Galaxy
        enable_merges = true;
        soft          = 0.10;

        const int    N_gal  = 200;
        const double R_disk = 3.0; // initial spread radius

        bodies.clear();
        q = num::Vector(2 * N_gal, 0.0);
        v = num::Vector(2 * N_gal, 0.0);

        std::mt19937                           rng(20240101);
        std::uniform_real_distribution<double> uni(0.0, 1.0);

        // Estimate total mass to seed tangential speeds
        // With masses uniform in [0.3, 2.0], mean ~= 1.15, M_total ~= 230
        const double M_total_est = N_gal * 1.15;

        for (int i = 0; i < N_gal; ++i) {
            // Radius: slightly concentrated toward center
            double r     = R_disk * std::pow(uni(rng), 0.65);
            double theta = uni(rng) * 2.0 * M_PI;

            // Mass: random uniform [0.3, 2.0]
            double m = 0.3 + 1.7 * uni(rng);

            // Tangential velocity: ~35% of circular velocity at this radius
            // This gives highly elliptical orbits that cross each other
            double vcirc = (r > 0.05) ? std::sqrt(G * M_total_est / r) : 0.0;
            double vtang = vcirc
                           * (0.25 + 0.20 * uni(rng)); // 25-45% of circular

            // Small random perturbation so orbits have different phases
            double vrand_x = (2.0 * uni(rng) - 1.0) * 0.3;
            double vrand_y = (2.0 * uni(rng) - 1.0) * 0.3;

            num::Vec2View pos{q}, vel{v};
            pos.x(i) = r * std::cos(theta);
            pos.y(i) = r * std::sin(theta);
            // Tangential direction (counterclockwise): (-sin(theta),
            // cos(theta))
            vel.x(i) = -vtang * std::sin(theta) + vrand_x;
            vel.y(i) = vtang * std::cos(theta) + vrand_y;

            bodies.push_back({m, disp_r(m), mass_color(m), phys_r(m)});
        }
    }

    E0 = total_energy();
}

} // namespace nbody
