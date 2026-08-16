/// @file 05_symplectic_nbody_ode.cpp
/// @brief RK4, Adaptive RK45, Verlet, and Yoshida 4th-Order Symplectic Integrators.
#include <numerics.hpp>
#include <iostream>
#include <cmath>

int main() {
    using namespace num;

    // Harmonic Oscillator: d/dt [q, p] = [p, -q]
    auto f = [](real t, const Vector& y, Vector& dy) {
        dy[0] = y[1];
        dy[1] = -y[0];
    };

    Vector y0{1.0, 0.0};
    ODEParams p;
    p.t0 = 0.0;
    p.tf = 10.0;
    p.h = 0.1;

    // 1. Classical RK4 Integrator
    auto rk4_sol = ode_rk4(f, y0, p);
    std::cout << "RK4 completed in " << rk4_sol.steps << " steps. Final state q(tf) = " << rk4_sol.u[0] << "\n";

    // 2. Adaptive Dormand-Prince RK45 Integrator
    p.rtol = 1e-6;
    p.atol = 1e-6;
    auto rk45_sol = ode_rk45(f, y0, p);
    std::cout << "RK45 Adaptive completed in " << rk45_sol.steps << " steps. Final state q(tf) = " << rk45_sol.u[0] << "\n";

    // 3. Yoshida 4th-Order Symplectic Integrator for Hamiltonian Systems
    auto accel = [](const Vector& q, Vector& a) {
        a[0] = -q[0];
    };

    Vector q0{1.0}, v0{0.0};
    auto yoshida_sol = ode_yoshida4(accel, q0, v0, p);
    real H0 = 0.5 * (v0[0]*v0[0] + q0[0]*q0[0]);
    real Hf = 0.5 * (yoshida_sol.v[0]*yoshida_sol.v[0] + yoshida_sol.q[0]*yoshida_sol.q[0]);
    std::cout << "Yoshida4 Symplectic Energy Error |Hf - H0| = " << std::abs(Hf - H0) << "\n";

    return 0;
}
