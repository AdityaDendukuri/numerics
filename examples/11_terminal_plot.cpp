/// @file 11_terminal_plot.cpp
/// @brief Plot a function in the terminal.
#include <cmath>
#include <numerics.hpp>

int main() {
    using namespace num;

    // Generate sample sine wave data
    std::vector<double> x, y;
    for (int i = 0; i <= 60; ++i) {
        double xi = i * 0.1;
        x.push_back(xi);
        y.push_back(std::sin(xi));
    }

    plt::plot(x, y, "sin(x)", "lines");
    plt::title("In-Terminal ASCII Waveform Plot");
    plt::xlabel("Time (x)");
    plt::ylabel("Amplitude sin(x)");
    plt::savefig("plot.png");

    return 0;
}
