/// @file 22_heat_equation.cpp
/// @brief Solve the heat equation on a periodic domain by the Fourier transform.
///
/// Each mode of u_t = diffusivity * u_xx decays by exp(-diffusivity * k^2 * t).
/// The inverse FFT returns an unnormalized result, so divide by the grid size.
#include <algorithm>
#include <array>
#include <cmath>
#include <io/json.hpp>
#include <numerics.hpp>
#include <string_view>
#include <utility>
#include <vector>

using namespace num;
using namespace num::spectral;

int main(int argc, char **argv) {
    const bool plot = argc > 1 && std::string_view(argv[1]) == "--plot";
    constexpr idx n = 256;
    constexpr real diffusivity = 0.08;
    std::vector<real> grid(n);
    vec<real> initial(n);
    for (idx i = 0; i < n; ++i) {
        const real x = 2.0 * pi * static_cast<real>(i) / n;
        const real distance = std::min(x, 2.0 * pi - x);
        grid[i] = x;
        initial[i] = 0.35 + std::exp(-0.5 * std::pow(distance / 0.18, 2))
                     + 0.3 * std::cos(3.0 * x);
    }

    vec<cplx> modes(n / 2 + 1);
    rfft(initial, modes);
    io::json results = io::json::array();
    for (const auto &[time, label] : std::array<std::pair<real, const char *>, 4>{{
             {0.0, "t = 0"}, {0.05, "t = 0.05"},
             {0.2, "t = 0.2"}, {0.8, "t = 0.8"}}}) {
        vec<cplx> damped = modes;
        for (idx k = 0; k < damped.size(); ++k)
            damped[k] *= std::exp(-diffusivity * static_cast<real>(k * k) * time);

        vec<real> solution(n);
        irfft(damped, static_cast<int>(n), solution);
        real mass = 0.0, peak = 0.0;
        std::vector<real> values(n);
        for (idx i = 0; i < n; ++i) {
            values[i] = solution[i] / static_cast<real>(n);
            mass += values[i] / static_cast<real>(n);
            peak = std::max(peak, values[i]);
        }
        results.push_back({{"time", time}, {"mean", mass}, {"peak", peak}});
        if (plot)
            plt::plot(grid, values, label, "lines");
    }
    io::write_json(results, "result.json");
    if (plot) {
        plt::title("Periodic heat equation with FFT");
        plt::xlabel("x");
        plt::ylabel("temperature");
        plt::legend();
        plt::savefig("plot.png");
    }
}
