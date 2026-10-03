/// @file 05_symplectic_nbody_ode.cpp
/// @brief Explicit, adaptive and symplectic integrators on the harmonic oscillator.
///
/// `ode_rk4` and `ode_rk45` integrate y' = f(t, y). `ode_verlet` and `ode_yoshida4` integrate q'' =
/// a(q), and because they are symplectic their energy error stays bounded over long times instead
/// of drifting. The oscillator q'' = -q has energy (q^2 + v^2)/2 = 1/2 and solution q(t) = cos t.
#include <cmath>
#include <cstdio>
#include <numerics.hpp>
#include <string_view>
#include <vector>

using namespace num;

int main(int argc, char **argv) {
    const bool plot = argc > 1 && std::string_view(argv[1]) == "--plot";
    constexpr real tf = 200.0;
    auto energy = [](real q, real v) { return 0.5 * ((q * q) + (v * v)); };

    const auto rhs = [](real, const vec<real> &y, vec<real> &dy) {
        dy[0] = y[1];
        dy[1] = -y[0];
    };
    const auto acceleration = [](const vec<real> &q, vec<real> &a) { a[0] = -q[0]; };

    std::printf("t = %.0f, step 0.1           q(t)        error      energy error  steps\n", tf);
    const ode_params fixed{.t0 = 0.0, .tf = tf, .h = 0.1};
    const ode_result rk4 = ode_rk4(rhs, vec<real>{1.0, 0.0}, fixed);
    std::printf("rk4                    %12.8f  %10.2e  %12.2e  %5zu\n", rk4.u[0],
                std::abs(rk4.u[0] - std::cos(tf)), energy(rk4.u[0], rk4.u[1]) - 0.5,
                static_cast<std::size_t>(rk4.steps));

    const ode_result rk45 = ode_rk45(rhs, vec<real>{1.0, 0.0},
                                     {.t0 = 0.0, .tf = tf, .h = 0.1, .rtol = 1e-8, .atol = 1e-10});
    std::printf("rk45 (rtol 1e-8)       %12.8f  %10.2e  %12.2e  %5zu\n", rk45.u[0],
                std::abs(rk45.u[0] - std::cos(tf)), energy(rk45.u[0], rk45.u[1]) - 0.5,
                static_cast<std::size_t>(rk45.steps));

    const symplectic_result verlet = ode_verlet(acceleration, vec<real>{1.0}, vec<real>{0.0}, fixed);
    std::printf("verlet                 %12.8f  %10.2e  %12.2e  %5zu\n", verlet.q[0],
                std::abs(verlet.q[0] - std::cos(tf)), energy(verlet.q[0], verlet.v[0]) - 0.5,
                static_cast<std::size_t>(verlet.steps));

    const symplectic_result yoshida =
        ode_yoshida4(acceleration, vec<real>{1.0}, vec<real>{0.0}, fixed);
    std::printf("yoshida4               %12.8f  %10.2e  %12.2e  %5zu\n", yoshida.q[0],
                std::abs(yoshida.q[0] - std::cos(tf)), energy(yoshida.q[0], yoshida.v[0]) - 0.5,
                static_cast<std::size_t>(yoshida.steps));

    if (plot) {
        std::vector<double> t, q, v;
        for (const auto &step : yoshida4(acceleration, vec<real>{1.0}, vec<real>{0.0},
                                         ode_params{.t0 = 0.0, .tf = 20.0, .h = 0.05})) {
            t.push_back(step.t);
            q.push_back(step.q[0]);
            v.push_back(step.v[0]);
        }
        plt::plot(t, q, "q(t)", "lines");
        plt::plot(t, v, "v(t)", "lines");
        plt::title("05 Yoshida 4th-order oscillator");
        plt::legend();
        plt::show();
    }
}
