/// @file 05_symplectic_nbody_ode.cpp
/// @brief RK4, Adaptive RK45, Verlet, and Yoshida 4th-Order Symplectic Integrators.
#include <iostream>
#include <numerics.hpp>
#include <vector>

int main() {
    using namespace num;

    auto accel = [](const Vector &q, Vector &a) { a[0] = -q[0]; };

    Vector q0{1.0}, v0{0.0};
    ODEParams p;
    p.t0 = 0.0;
    p.tf = 10.0;
    p.h = 0.05;

    std::vector<double> t_vec, q_vec, v_vec;
    auto yoshida_stepper = yoshida4(accel, q0, v0, p);
    for (const auto &step : yoshida_stepper) {
        t_vec.push_back(step.t);
        q_vec.push_back(step.q[0]);
        v_vec.push_back(step.v[0]);
    }

    std::cout << "Yoshida 4th-Order Symplectic Integration completed in " << t_vec.size()
              << " steps.\n";

    plt::plot(t_vec, q_vec, "Position q(t)", "lines");
    plt::plot(t_vec, v_vec, "Velocity v(t)", "lines");
    plt::title("05 Symplectic ODE: Yoshida 4th-Order Harmonic Oscillator");
    plt::xlabel("Time t");
    plt::ylabel("Phase Coordinates");
    plt::legend();
    plt::show_dumb(140, 35);

    return 0;
}
