/// @file plot/gnuplot.hpp
/// @brief Plot data helpers and a low-level owning gnuplot pipe.
#pragma once

#include "container/vector.hpp"
#include <cstdio>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace num {

using Point = std::pair<double, double>;

/// Ordered (x,y) samples accepted by the plotting helpers.
struct Series : std::vector<Point> {
    using std::vector<Point>::vector;
    /// Append one sample.
    void store(double x, double y) { emplace_back(x, y); }
};

/// Extract one row as (x_j,u_j) plot data.
inline Series row_slice(const Vector &u, int N, double h, int row) {
    Series result;
    result.reserve(static_cast<std::size_t>(N));
    for (int column = 0; column < N; ++column) {
        result.store((column + 1) * h, u[(static_cast<std::size_t>(row) * N) + column]);
    }
    return result;
}

/// Extract one column as (y_i,u_i) plot data.
inline Series col_slice(const Vector &u, int N, double h, int column) {
    Series result;
    result.reserve(static_cast<std::size_t>(N));
    for (int row = 0; row < N; ++row) {
        result.store((row + 1) * h, u[(static_cast<std::size_t>(row) * N) + column]);
    }
    return result;
}

/// Extract one row from a field exposing vec(), N(), and h().
template <class Field>
inline Series row_slice(const Field &field, int row) {
    return row_slice(field.vec(), field.N(), field.h(), row);
}

/// Extract one column from a field exposing vec(), N(), and h().
template <class Field>
inline Series col_slice(const Field &field, int column) {
    return col_slice(field.vec(), field.N(), field.h(), column);
}

/// Owning pipe for issuing low-level commands directly to gnuplot.
class Gnuplot {
  public:
    /// Launch gnuplot with optional command-line arguments.
    explicit Gnuplot(const std::string &args = "") {
        const std::string command = "gnuplot " + args;
        pipe_ = popen(command.c_str(), "w");
        if (!pipe_) {
            throw std::runtime_error("could not open gnuplot -- is it installed?");
        }
    }

    ~Gnuplot() {
        if (pipe_) {
            pclose(pipe_);
        }
    }

    Gnuplot(const Gnuplot &) = delete;
    Gnuplot &operator=(const Gnuplot &) = delete;

    /// Send a raw gnuplot command.
    Gnuplot &operator<<(const std::string &command) {
        fputs(command.c_str(), pipe_);
        return *this;
    }

    /// Send one inline two-column data block and flush it.
    void send1d(const Series &data) {
        for (const auto &[x, y] : data) {
            fprintf(pipe_, "%.15g %.15g\n", x, y);
        }
        fputs("e\n", pipe_);
        fflush(pipe_);
    }

    /// Flush pending commands to gnuplot.
    void flush() { fflush(pipe_); }

  private:
    FILE *pipe_ = nullptr;
};

/// Apply SIAM-style line, grid, border, and legend settings.
inline void apply_siam_style(Gnuplot &gp) {
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

/// Apply GitHub Primer Light style (clean white canvas, primer borders and accents).
inline void apply_github_light_style(Gnuplot &gp) {
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

/// Apply GitHub Primer Dark style.
inline void apply_github_dark_style(Gnuplot &gp) {
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

/// Configure logarithmic x and y axes on a raw pipe.
inline void set_loglog(Gnuplot &gp) {
    gp << "set logscale xy\nset format x '10^{%L}'\nset format y '10^{%L}'\n";
}

/// Configure a logarithmic x axis on a raw pipe.
inline void set_logx(Gnuplot &gp) {
    gp << "set logscale x\nset format x '10^{%L}'\n";
}

/// Configure PNG output on a raw pipe.
inline void save_png(Gnuplot &gp, const std::string &filename, int width = 900, int height = 600) {
    gp << "set terminal pngcairo size " + std::to_string(width) + "," + std::to_string(height) +
              " enhanced font 'Arial,11'\n"
       << "set output '" + filename + "'\n";
}

} // namespace num
