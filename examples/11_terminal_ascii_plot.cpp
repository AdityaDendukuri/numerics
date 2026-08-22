/// @file 11_terminal_ascii_plot.cpp
/// @brief Quick-start guide for in-terminal ASCII plotting via gnuplot set terminal dumb.
#include <cmath>
#include <iostream>
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

    std::cout << "Rendering In-Terminal ASCII Plot (set terminal dumb size 120, 30)...\n\n";

    // 1. Matplotlib-style plotting API rendered directly in terminal ASCII
    plt::plot(x, y, "sin(x)", "lines");
    plt::title("In-Terminal ASCII Waveform Plot");
    plt::xlabel("Time (x)");
    plt::ylabel("Amplitude sin(x)");
    plt::show_dumb(120, 25);

    return 0;
}
