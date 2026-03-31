/// @file plot/plot.hpp
/// @brief Matplotlib-style plotting via a gnuplot pipe.
///
/// Requires gnuplot in PATH (runtime only -- no link-time dependency).
///
/// ### Quick start
/// @code
///   #include "numerics.hpp"
///
///   num::Vector t = ..., x = ...;
///
///   num::plt::plot(t, x, "signal");
///   num::plt::title("Damped oscillator");
///   num::plt::xlabel("t");
///   num::plt::ylabel("x(t)");
///   num::plt::show();          // opens gnuplot window; blocks until closed
///
///   num::plt::savefig("out.png");   // alternative: save to PNG
/// @endcode
///
/// Multiple series on one figure:
/// @code
///   num::plt::plot(t, x, "Verlet");
///   num::plt::plot(t, y, "RK4");
///   num::plt::legend();
///   num::plt::show();
/// @endcode
#pragma once

#include <cstdio>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace num {

/// (x, y) data point.
using Point = std::pair<double, double>;

/// Ordered (x, y) series.
struct Series : std::vector<Point> {
    using std::vector<Point>::vector;
    /// Append a point to the series.
    void store(double x, double y) { emplace_back(x, y); }
};

// Low-level gnuplot pipe (used by bench_plot and plt:: internally)

/// Thin C++ wrapper around a gnuplot pipe (popen).
/// Commands via operator<<; inline data via send1d().
class Gnuplot {
public:
    explicit Gnuplot(const std::string& args = "") {
        std::string cmd = "gnuplot " + args;
        pipe_ = popen(cmd.c_str(), "w");
        if (!pipe_)
            throw std::runtime_error("could not open gnuplot -- is it installed?");
    }
    ~Gnuplot() { if (pipe_) pclose(pipe_); }
    Gnuplot(const Gnuplot&)            = delete;
    Gnuplot& operator=(const Gnuplot&) = delete;

    Gnuplot& operator<<(const std::string& cmd) { fputs(cmd.c_str(), pipe_); return *this; }
    void send1d(const Series& data) {
        for (const auto& [x, y] : data) fprintf(pipe_, "%.15g %.15g\n", x, y);
        fputs("e\n", pipe_);
        fflush(pipe_);
    }
    void flush() { fflush(pipe_); }

private:
    FILE* pipe_ = nullptr;
};

/// Apply SIAM-style theme to a raw Gnuplot pipe.
inline void apply_siam_style(Gnuplot& gp) {
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

inline void set_loglog(Gnuplot& gp) {
    gp << "set logscale xy\nset format x '10^{%L}'\nset format y '10^{%L}'\n";
}
inline void set_logx(Gnuplot& gp) {
    gp << "set logscale x\nset format x '10^{%L}'\n";
}
inline void save_png(Gnuplot& gp, const std::string& filename, int w = 900, int h = 600) {
    gp << "set terminal pngcairo size " + std::to_string(w) + "," + std::to_string(h)
                                        + " enhanced font 'Arial,11'\n"
       << "set output '" + filename + "'\n";
}

namespace plt {
namespace detail {

struct SeriesEntry {
    Series      data;
    std::string label;
    std::string style;  // gnuplot "with" clause, e.g. "lines"
};

struct Panel {
    std::vector<SeriesEntry> series;
    std::string title_, xlabel_, ylabel_;
    std::string xrange_, yrange_;
    bool legend_ = false;
    bool logx_   = false;
    bool logy_   = false;
};

struct State {
    Panel              current;
    std::vector<Panel> panels;      // accumulated panels in multiplot mode
    int mp_rows_ = 0, mp_cols_ = 0; // 0 = single-plot mode

    void reset() { *this = State{}; }
};

inline State& state() { static State s; return s; }

// Write all datablocks for a panel, then emit the plot command for that panel.
// block_offset: index of the first $d_N block allocated to this panel.
inline void write_panel(FILE* pipe, const Panel& p, int block_offset) {
    if (p.series.empty()) return;

    // Labels / decorators
    if (!p.title_.empty())  fprintf(pipe, "set title '%s'\n",  p.title_.c_str());
    else                    fputs("unset title\n", pipe);
    if (!p.xlabel_.empty()) fprintf(pipe, "set xlabel '%s'\n", p.xlabel_.c_str());
    else                    fputs("unset xlabel\n", pipe);
    if (!p.ylabel_.empty()) fprintf(pipe, "set ylabel '%s'\n", p.ylabel_.c_str());
    else                    fputs("unset ylabel\n", pipe);
    if (!p.xrange_.empty()) fprintf(pipe, "set xrange %s\n",   p.xrange_.c_str());
    else                    fputs("set xrange [*:*]\n", pipe);
    if (!p.yrange_.empty()) fprintf(pipe, "set yrange %s\n",   p.yrange_.c_str());
    else                    fputs("set yrange [*:*]\n", pipe);

    if (p.logx_ && p.logy_) {
        fputs("set logscale xy\nset format x '10^{%L}'\nset format y '10^{%L}'\n", pipe);
    } else if (p.logx_) {
        fputs("set logscale x\nset format x '10^{%L}'\n", pipe);
    } else if (p.logy_) {
        fputs("set logscale y\nset format y '10^{%L}'\n", pipe);
    } else {
        fputs("unset logscale\n", pipe);
    }

    if (p.legend_) {
        fputs("set key top right Left reverse samplen 3 spacing 1.2\nset key box lt 1 lw 0.5\n", pipe);
    } else {
        fputs("unset key\n", pipe);
    }

    // plot command — references pre-written datablocks
    fputs("plot ", pipe);
    for (std::size_t i = 0; i < p.series.size(); ++i) {
        if (i) fputs(", ", pipe);
        const auto& e = p.series[i];
        fprintf(pipe, "$d_%d with %s ls %zu",
                block_offset + (int)i, e.style.c_str(), i + 1);
        if (!e.label.empty()) fprintf(pipe, " title '%s'", e.label.c_str());
        else                  fputs(" notitle", pipe);
    }
    fputc('\n', pipe);
}

inline void flush_to(FILE* pipe, const std::string& outfile) {
    auto& s = state();

    // Collect all panels (push current last)
    std::vector<Panel> all = s.panels;
    all.push_back(s.current);

    bool multiplot = (s.mp_rows_ > 0);

    // Terminal
    if (outfile.empty()) {
        int h = multiplot ? 300 * s.mp_rows_ : 600;
        fprintf(pipe, "set terminal qt size 900,%d\n", h);
    } else {
        std::string ext = outfile.size() > 4 ? outfile.substr(outfile.size() - 4) : "";
        if (ext == ".pdf") {
            double h = multiplot ? 3.0 * s.mp_rows_ : 4.0;
            fprintf(pipe, "set terminal pdfcairo size 6,%.0f font 'Arial,11'\n", h);
        } else {
            int h = multiplot ? 350 * s.mp_rows_ : 600;
            fprintf(pipe, "set terminal pngcairo size 900,%d enhanced font 'Arial,11'\n", h);
        }
        fprintf(pipe, "set output '%s'\n", outfile.c_str());
    }

    // Global theme
    fputs("set style line 1 lt 1 lw 2 pt 7  ps 0.7 lc rgb '#2c3e50'\n", pipe);
    fputs("set style line 2 lt 2 lw 2 pt 5  ps 0.7 lc rgb '#c0392b'\n", pipe);
    fputs("set style line 3 lt 3 lw 2 pt 9  ps 0.7 lc rgb '#2980b9'\n", pipe);
    fputs("set style line 4 lt 4 lw 2 pt 13 ps 0.7 lc rgb '#27ae60'\n", pipe);
    fputs("set style line 5 lt 5 lw 2 pt 11 ps 0.7 lc rgb '#8e44ad'\n", pipe);
    fputs("set style line 100 lt 1 lw 0.5 lc rgb '#cccccc'\n", pipe);
    fputs("set grid back ls 100\n", pipe);
    fputs("set border 3 lw 1.5\n", pipe);
    fputs("set tics nomirror\n", pipe);

    // Write all datablocks up front (required for multiplot; harmless for single)
    int block = 0;
    for (const auto& p : all)
        for (const auto& e : p.series) {
            fprintf(pipe, "$d_%d << EOD\n", block++);
            for (const auto& [x, y] : e.data)
                fprintf(pipe, "%.15g %.15g\n", x, y);
            fputs("EOD\n", pipe);
        }

    if (multiplot) {
        fprintf(pipe, "set multiplot layout %d,%d spacing 0.08,0.12\n",
                s.mp_rows_, s.mp_cols_);
        int off = 0;
        for (const auto& p : all) {
            write_panel(pipe, p, off);
            off += (int)p.series.size();
        }
        fputs("unset multiplot\n", pipe);
    } else {
        write_panel(pipe, all.back(), 0);
    }

    fflush(pipe);
}

} // namespace detail

// -- Series builders ----------------------------------------------------------

/// Append a Series (vector of (x,y) pairs) to the current panel.
inline void plot(const Series& data,
                 const std::string& label = "",
                 const std::string& style = "lines") {
    detail::state().current.series.push_back({data, label, style});
}

/// Append parallel x and y vectors to the current panel.
inline void plot(const std::vector<double>& x,
                 const std::vector<double>& y,
                 const std::string& label = "",
                 const std::string& style = "lines") {
    Series s;
    s.reserve(x.size());
    for (std::size_t i = 0; i < x.size() && i < y.size(); ++i)
        s.emplace_back(x[i], y[i]);
    detail::state().current.series.push_back({std::move(s), label, style});
}

// -- Decorators ---------------------------------------------------------------

inline void title(const std::string& t)  { detail::state().current.title_  = t; }
inline void xlabel(const std::string& l) { detail::state().current.xlabel_ = l; }
inline void ylabel(const std::string& l) { detail::state().current.ylabel_ = l; }

/// Set x-axis range, e.g. xlim(0, 10).
inline void xlim(double lo, double hi) {
    detail::state().current.xrange_ = "[" + std::to_string(lo) + ":" + std::to_string(hi) + "]";
}
/// Set y-axis range.
inline void ylim(double lo, double hi) {
    detail::state().current.yrange_ = "[" + std::to_string(lo) + ":" + std::to_string(hi) + "]";
}

/// Show a legend using the labels passed to plot().
inline void legend() { detail::state().current.legend_ = true; }

/// Log-log axes.
inline void loglog()  { detail::state().current.logx_ = detail::state().current.logy_ = true; }
/// Log y-axis only.
inline void semilogy() { detail::state().current.logy_ = true; }
/// Log x-axis only.
inline void semilogx() { detail::state().current.logx_ = true; }

// -- Multiplot ----------------------------------------------------------------

/// Start a multiplot with the given grid dimensions.
/// Call next() to advance panels, then savefig()/show() to emit everything.
/// @code
///   num::plt::subplot(2, 1);
///   num::plt::plot(a);  num::plt::xlabel("x");
///   num::plt::next();
///   num::plt::plot(b);  num::plt::xlabel("y");
///   num::plt::savefig("out.png");
/// @endcode
inline void subplot(int rows, int cols = 1) {
    detail::state().reset();
    detail::state().mp_rows_ = rows;
    detail::state().mp_cols_ = cols;
}

/// Advance to the next panel (commits the current panel, starts a fresh one).
inline void next() {
    detail::state().panels.push_back(detail::state().current);
    detail::state().current = detail::Panel{};
}

// -- Output -------------------------------------------------------------------

/// Open an interactive gnuplot window; blocks until the window is closed.
/// Resets figure state afterwards.
inline void show() {
    FILE* pipe = popen("gnuplot", "w");
    if (!pipe) throw std::runtime_error("could not open gnuplot -- is it installed?");
    detail::flush_to(pipe, "");
    fputs("pause mouse close\n", pipe);
    fflush(pipe);
    pclose(pipe);
    detail::state().reset();
}

/// Save the figure to a file (PNG or PDF inferred from extension).
/// Resets figure state afterwards.
inline void savefig(const std::string& filename) {
    FILE* pipe = popen("gnuplot", "w");
    if (!pipe) throw std::runtime_error("could not open gnuplot -- is it installed?");
    detail::flush_to(pipe, filename);
    fflush(pipe);
    pclose(pipe);
    detail::state().reset();
}

/// Clear the current figure (discard all series and settings).
inline void clf() { detail::state().reset(); }

} // namespace plt
} // namespace num
