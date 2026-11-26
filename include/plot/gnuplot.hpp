/// @file plot/gnuplot.hpp
/// @brief Plot data helpers and a raw gnuplot pipe.
#pragma once

#include "core/types.hpp"
#include "container/vector.hpp"
#include <cstdio>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace num {

/// @brief Point of a plot: \f$(x, y)\f$.
using plot_point = std::pair<double, double>;

/// @brief Series of points of a plot, in order.
struct series : array<plot_point> {
    using array<plot_point>::vector;
    /// @brief Append the point \f$(x, y)\f$.
    void store(double x, double y) { emplace_back(x, y); }
};

/// @brief Take a row of a field as a series: \f$(x_j, u_{rj})\f$.
inline series row_slice(const vec<real> &u, int N, double h, int row) {
    series result;
    result.reserve(static_cast<std::size_t>(N));
    for (int column = 0; column < N; ++column) {
        result.store((column + 1) * h, u[(static_cast<std::size_t>(row) * N) + column]);
    }
    return result;
}

/// @brief Take a column of a field as a series: \f$(y_i, u_{ic})\f$.
inline series col_slice(const vec<real> &u, int N, double h, int column) {
    series result;
    result.reserve(static_cast<std::size_t>(N));
    for (int row = 0; row < N; ++row) {
        result.store((row + 1) * h, u[(static_cast<std::size_t>(row) * N) + column]);
    }
    return result;
}

/// @brief Row of a field.
template <class Field>
inline series row_slice(const Field &field, int row) {
    return row_slice(field.as_vec(), field.N(), field.h(), row);
}

/// @brief Column of a field.
template <class Field>
inline series col_slice(const Field &field, int column) {
    return col_slice(field.as_vec(), field.N(), field.h(), column);
}

/// @brief Pipe to a gnuplot process, for sending commands.
class gnuplot {
  public:
    /// @brief Start gnuplot with `args`.
    explicit gnuplot(const std::string &args = "") {
        const std::string command = "gnuplot " + args;
        pipe_ = popen(command.c_str(), "w");
        if (!pipe_) {
            throw std::runtime_error("could not open gnuplot -- is it installed?");
        }
    }

    ~gnuplot() {
        if (pipe_) {
            pclose(pipe_);
        }
    }

    gnuplot(const gnuplot &) = delete;
    gnuplot &operator=(const gnuplot &) = delete;

    /// @brief Send a command.
    gnuplot &operator<<(const std::string &command) {
        fputs(command.c_str(), pipe_);
        return *this;
    }

    /// @brief Send a block of two-column data.
    void send1d(const series &data) {
        for (const auto &[x, y] : data) {
            fprintf(pipe_, "%.15g %.15g\n", x, y);
        }
        fputs("e\n", pipe_);
        fflush(pipe_);
    }

    /// @brief Flush the commands sent so far.
    void flush() { fflush(pipe_); }

  private:
    FILE *pipe_ = nullptr;
};

/// @brief Set the SIAM style: lines, grid, border, legend.
inline void apply_siam_style(gnuplot &gp) {
    gp << "set style line 1 lt 1 lw 2 pt 7  ps 0.8 lc rgb 'black'\n"
       << "set style line 2 lt 2 lw 2 pt 5  ps 0.8 lc rgb 'black'\n"
       << "set style line 3 lt 3 lw 2 pt 9  ps 0.8 lc rgb 'black'\n"
       << "set style line 4 lt 4 lw 2 pt 13 ps 0.8 lc rgb 'black'\n"
       << "set style line 5 lt 5 lw 2 pt 11 ps 0.8 lc rgb 'black'\n"
       << "set style line 6 lt 6 lw 2 pt 15 ps 0.8 lc rgb 'black'\n"
       << "set style line 100 lt 1 lw 0.5 lc rgb '#cccccc'\n"
       << "set grid back ls 100\n"
       << "set border 3 lw 1.5\n"
       << "set tics nomirror\n"
       << "set key top left Left reverse samplen 3 spacing 1.2\n"
       << "set key box lt 1 lw 0.5\n";
}

/// @brief Set the GitHub Primer light style.
inline void apply_github_light_style(gnuplot &gp) {
    gp << "set style line 1 lt 1 lw 2 pt 7  ps 0.75 lc rgb '#0969da'\n"
       << "set style line 2 lt 2 lw 2 pt 5  ps 0.75 lc rgb '#1a7f37'\n"
       << "set style line 3 lt 3 lw 2 pt 9  ps 0.75 lc rgb '#cf222e'\n"
       << "set style line 4 lt 4 lw 2 pt 13 ps 0.75 lc rgb '#8250df'\n"
       << "set style line 5 lt 5 lw 2 pt 11 ps 0.75 lc rgb '#bf8700'\n"
       << "set style line 6 lt 6 lw 2 pt 15 ps 0.75 lc rgb '#0598bc'\n"
       << "set style line 7 lt 7 lw 2 pt 4  ps 0.75 lc rgb '#bf3989'\n"
       << "set style line 8 lt 8 lw 2 pt 6  ps 0.75 lc rgb '#656d76'\n"
       << "set style line 100 lt 1 lw 0.5 lc rgb '#eaeef2'\n"
       << "set grid back ls 100\n"
       << "set border 3 lw 1.2 lc rgb '#d0d7de'\n"
       << "set tics nomirror textcolor rgb '#1f2328'\n"
       << "set key top left Left reverse samplen 3 spacing 1.2\n"
       << "set key box lt 1 lc rgb '#d0d7de' lw 0.8\n"
       << "set key textcolor rgb '#1f2328'\n";
}

/// @brief Set the GitHub Primer dark style.
inline void apply_github_dark_style(gnuplot &gp) {
    gp << "set style line 1 lt 1 lw 2 pt 7  ps 0.75 lc rgb '#58a6ff'\n"
       << "set style line 2 lt 2 lw 2 pt 5  ps 0.75 lc rgb '#3fb950'\n"
       << "set style line 3 lt 3 lw 2 pt 9  ps 0.75 lc rgb '#f85149'\n"
       << "set style line 4 lt 4 lw 2 pt 13 ps 0.75 lc rgb '#bc8cff'\n"
       << "set style line 5 lt 5 lw 2 pt 11 ps 0.75 lc rgb '#d29922'\n"
       << "set style line 6 lt 6 lw 2 pt 15 ps 0.75 lc rgb '#39c5cf'\n"
       << "set style line 7 lt 7 lw 2 pt 4  ps 0.75 lc rgb '#f778ba'\n"
       << "set style line 8 lt 8 lw 2 pt 6  ps 0.75 lc rgb '#8b949e'\n"
       << "set style line 100 lt 1 lw 0.5 lc rgb '#30363d'\n"
       << "set grid back ls 100\n"
       << "set border 3 lw 1.2 lc rgb '#30363d'\n"
       << "set tics nomirror textcolor rgb '#e6edf3'\n"
       << "set key top left Left reverse samplen 3 spacing 1.2\n"
       << "set key box lt 1 lc rgb '#30363d' lw 0.8\n"
       << "set key textcolor rgb '#e6edf3'\n";
}

/// @brief Use logarithmic x and y axes.
inline void set_loglog(gnuplot &gp) {
    gp << "set logscale xy\nset format x '10^{%L}'\nset format y '10^{%L}'\n";
}

/// @brief Use a logarithmic x axis.
inline void set_logx(gnuplot &gp) {
    gp << "set logscale x\nset format x '10^{%L}'\n";
}

/// @brief Write a PNG of `width x height`.
inline void save_png(gnuplot &gp, const std::string &filename, int width = 900, int height = 600) {
    gp << "set terminal pngcairo size " + std::to_string(width) + "," + std::to_string(height) +
              " enhanced font 'Arial,11'\n"
       << "set output '" + filename + "'\n";
}

} // namespace num
