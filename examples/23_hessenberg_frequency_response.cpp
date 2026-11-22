/// @file 23_hessenberg_frequency_response.cpp
/// @brief Plot resonances with one Hessenberg reduction and many complex shifts.
///
/// A chain of three coupled oscillators responds to a periodic input. The real
/// part of the shift controls damping; the imaginary part is the input frequency.
#include <array>
#include <cmath>
#include <complex>
#include <io/json.hpp>
#include <numerics.hpp>
#include <string_view>
#include <utility>
#include <vector>

using namespace num;

int main(int argc, char **argv) {
    const bool plot = argc > 1 && std::string_view(argv[1]) == "--plot";
    mat<real> A(6, 6, 0.0);
    for (idx j = 0; j < 3; ++j) {
        const real frequency = static_cast<real>(j + 1);
        A(2 * j, 2 * j) = A(2 * j + 1, 2 * j + 1) = -0.04;
        A(2 * j, 2 * j + 1) = -frequency;
        A(2 * j + 1, 2 * j) = frequency;
        if (j > 0) {
            A(2 * j, 2 * (j - 1)) = 0.8;
            A(2 * j + 1, 2 * (j - 1) + 1) = 0.8;
        }
    }
    vec<real> input(6, 0.0);
    input[0] = 1.0;
    const hessenberg_resolvent reduced(A);

    constexpr idx samples = 500;
    std::vector<real> frequency(samples);
    for (idx i = 0; i < samples; ++i)
        frequency[i] = 4.0 * static_cast<real>(i) / static_cast<real>(samples - 1);

    io::json results = io::json::array();
    for (const auto &[damping, label] : std::array<std::pair<real, const char *>, 3>{{
             {0.02, "Re(s) = 0.02"}, {0.12, "Re(s) = 0.12"}, {0.35, "Re(s) = 0.35"}}}) {
        array<cplx> shifts(samples);
        for (idx i = 0; i < samples; ++i)
            shifts[i] = cplx(damping, frequency[i]);
        const auto solutions = solve_batch(reduced, shifts, input);

        std::vector<real> gain(samples);
        idx peak = 0;
        for (idx i = 0; i < samples; ++i) {
            gain[i] = std::abs(solutions[i][4]);
            if (gain[i] > gain[peak])
                peak = i;
        }
        results.push_back({{"shift_real_part", damping},
                           {"peak_frequency", frequency[peak]}, {"peak_gain", gain[peak]}});
        if (plot)
            plt::plot(frequency, gain, label, "lines");
    }
    io::write_json(results, "result.json");
    if (plot) {
        plt::title("Resonances from one Hessenberg reduction");
        plt::xlabel("input frequency");
        plt::ylabel("response at the last oscillator");
        plt::semilogy();
        plt::legend();
        plt::savefig("plot.png");
    }
}
