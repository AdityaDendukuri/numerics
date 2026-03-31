/// @file include/nbody/sim.hpp
/// @brief Gravitational N-body simulation using num::ode_verlet /
/// num::ode_rk4_2nd.
///
/// NBodySim wraps the ODE module to provide a per-frame step() interface.
/// Each call to step(dt) runs exactly one time step of the chosen integrator:
///
///   Verlet  (symplectic, default) -- ode_verlet:    O(h^2), bounded energy
///   error RK4     (non-symplectic)      -- ode_rk4_2nd:   O(h^4), secular
///   energy drift
///
/// Both integrators share the same AccelFn / SymplecticResult interface, so
/// switching between them requires no restructuring of the state.
///
/// Positions and velocities are stored as flat num::Vector with stride 2
/// ([x0,y0, x1,y1, ...]).  Use num::Vec2View for readable index-free access.
#pragma once

#include "numerics.hpp"
#include <vector>
#include <utility>
#include <cstdint>
#include <cmath>
#include <string>

namespace nbody {

using num::real;
using num::Vector;

// Body descriptor

struct Body {
    double   mass;
    float    display_radius;
    uint32_t color; ///< RGBA packed (raylib Color layout)
    float    phys_radius =
        0.0f;       ///< physics-unit radius for merge detection (0 = disabled)
};

// Scenarios

enum class Scenario {
    Figure8, ///< Chenciner-Montgomery figure-8 choreography (3 equal masses)
    SolarSystem, ///< Sun + 4 planets on circular Keplerian orbits
    BinaryPlus,  ///< Equal-mass binary + one test particle on a wide orbit
    Galaxy,      ///< Random N bodies -- gravity only, bodies merge on contact
};

inline const char* scenario_name(Scenario s) {
    switch (s) {
        case Scenario::Figure8:
            return "Figure-8 Choreography";
        case Scenario::SolarSystem:
            return "Keplerian Orbits";
        case Scenario::BinaryPlus:
            return "Binary + Test Particle";
        case Scenario::Galaxy:
            return "Galaxy Collapse";
    }
    return "?";
}

// Color mapped from body mass: blue-white (low) to yellow (mid) to orange-red
// (high)
uint32_t mass_color(double m);

// Display radius (pixels) and physics radius for a given mass
float disp_r(double m);
float phys_r(double m);

// Simulation

struct NBodySim {
    std::vector<Body> bodies;
    Vector            q; ///< Positions: [x0,y0, x1,y1, ...]
    Vector            v; ///< Velocities: [vx0,vy0, vx1,vy1, ...]
    double            t    = 0.0;
    double            G    = 1.0;
    double            E0   = 0.0; ///< Initial total energy (for drift tracking)
    double            soft = 1e-3; ///< Softening length (avoids singularities)

    enum class Integrator { Verlet, Yoshida4, RK4 };
    Integrator integrator = Integrator::Verlet;
    bool       enable_merges =
        false; ///< true = check_merges() is active (Galaxy scenario)

    /// Reset to a preset scenario.
    void reset(Scenario s);

    /// Advance by exactly one time step dt.
    void step(double dt);

    /// Check all pairs for contact; merge overlapping bodies.
    /// Returns the list of (removed_idx, swapped_from_idx) operations in order
    /// so the caller can mirror them on any parallel index array (e.g. trails).
    std::vector<std::pair<int, int>> check_merges();

    int n() const {
        return static_cast<int>(bodies.size());
    }

    double kinetic_energy() const;
    double potential_energy() const;
    double total_energy() const {
        return kinetic_energy() + potential_energy();
    }

    /// Relative energy drift since reset: (E - E0) / |E0|
    double energy_drift() const {
        return E0 != 0.0 ? (total_energy() - E0) / std::abs(E0) : 0.0;
    }

    /// Fill acc from current positions (gravitational acceleration on each
    /// body).
    void make_accel(const Vector& pos, Vector& acc) const;
};

} // namespace nbody
