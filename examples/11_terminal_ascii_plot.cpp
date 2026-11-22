/// @file 11_terminal_ascii_plot.cpp
/// @brief Plotting in the terminal with gnuplot.
///
/// `plt::plot`, `plt::title` and the label calls build a figure as in matplotlib.
/// `plt::show_dumb(width, height)` draws it as text with gnuplot's dumb terminal, which works over
/// ssh and in logs. `plt::show` opens a window instead.
#include <cmath>
#include <numerics.hpp>
#include <vector>

int main() {
    using namespace num;

    std::vector<double> x, y;
    for (int i = 0; i <= 60; ++i) {
        x.push_back(i * 0.1);
        y.push_back(std::sin(i * 0.1));
    }

    plt::plot(x, y, "sin(x)", "lines");
    plt::title("sin(x) in the terminal");
    plt::xlabel("x");
    plt::ylabel("sin(x)");
    plt::show_dumb(120, 25);
}
