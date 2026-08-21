/// @file plot/plot.hpp
/// @brief Matplotlib-style plotting via a gnuplot pipe with ASCII terminal support.
#pragma once

#include "plot/gnuplot.hpp"
#include <algorithm>
#include <cstdio>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace num::plt {
namespace detail {

struct SeriesEntry {
  Series data;
  std::string label;
  std::string style; // gnuplot "with" clause, e.g. "lines"
};

/// 2-D field snapshot for heatmap rendering via gnuplot pm3d map.
struct HeatmapEntry {
  std::vector<double> data; // NxN row-major values
  int N = 0;
  double h = 1.0;
  double vmin = 0.0;
  double vmax = 1.0;
};

struct Panel {
  std::vector<SeriesEntry> series;
  std::vector<HeatmapEntry> heatmaps;
  std::string title_, xlabel_, ylabel_;
  std::string xrange_, yrange_;
  std::string palette_; // gnuplot palette string; empty = hot/fire
  bool legend_ = false;
  bool logx_ = false;
  bool logy_ = false;
};

struct State {
  Panel current;
  std::vector<Panel> panels; // accumulated panels in multiplot mode
  int mp_rows_ = 0, mp_cols_ = 0; // 0 = single-plot mode
  std::string term_override_ = ""; // "dumb", "qt", "pngcairo"
  int term_w_ = 140, term_h_ = 35;

  void reset() { *this = State{}; }
};

inline State& state() {
  static State s;
  return s;
}

// Write all datablocks for a panel, then emit the plot command for that panel.
inline void write_panel(FILE* pipe, const Panel& p, int block_offset) {
  if (p.series.empty() && p.heatmaps.empty()) {
    return;
  }

  // Common decorators
  if (!p.title_.empty()) {
    fprintf(pipe, "set title '%s'\n", p.title_.c_str());
  } else {
    fputs("unset title\n", pipe);
  }
  if (!p.xlabel_.empty()) {
    fprintf(pipe, "set xlabel '%s'\n", p.xlabel_.c_str());
  } else {
    fputs("unset xlabel\n", pipe);
  }
  if (!p.ylabel_.empty()) {
    fprintf(pipe, "set ylabel '%s'\n", p.ylabel_.c_str());
  } else {
    fputs("unset ylabel\n", pipe);
  }

  if (!p.heatmaps.empty()) {
    const auto& hm = p.heatmaps[0];
    if (!p.palette_.empty()) {
      fprintf(pipe, "set palette %s\n", p.palette_.c_str());
    } else {
      fputs("set palette defined "
            "(0 'white', 0.35 '#ffffb2', 0.65 '#fd8d3c', 1 '#bd0026')\n",
            pipe);
    }
    fprintf(pipe, "set cbrange [%g:%g]\n", hm.vmin, hm.vmax);
    fputs("set pm3d map\n", pipe);
    fputs("set size ratio 1\n", pipe);
    if (!p.xrange_.empty()) {
      fprintf(pipe, "set xrange %s\n", p.xrange_.c_str());
    } else {
      fputs("set xrange [*:*]\n", pipe);
    }
    if (!p.yrange_.empty()) {
      fprintf(pipe, "set yrange %s\n", p.yrange_.c_str());
    } else {
      fputs("set yrange [*:*]\n", pipe);
    }
    fputs("unset key\n", pipe);
    fprintf(pipe, "splot $d_%d with pm3d notitle\n", block_offset);
  } else {
    // Line-plot panel
    fputs("unset pm3d\n", pipe);
    if (!p.xrange_.empty()) {
      fprintf(pipe, "set xrange %s\n", p.xrange_.c_str());
    } else {
      fputs("set xrange [*:*]\n", pipe);
    }
    if (!p.yrange_.empty()) {
      fprintf(pipe, "set yrange %s\n", p.yrange_.c_str());
    } else {
      fputs("set yrange [*:*]\n", pipe);
    }

    if (p.logx_ && p.logy_) {
      fputs("set logscale xy\nset format x '10^{%L}'\nset format y "
            "'10^{%L}'\n",
            pipe);
    } else if (p.logx_) {
      fputs("set logscale x\nset format x '10^{%L}'\n", pipe);
    } else if (p.logy_) {
      fputs("set logscale y\nset format y '10^{%L}'\n", pipe);
    } else {
      fputs("unset logscale\n", pipe);
    }

    if (p.legend_) {
      fputs("set key top right Left reverse samplen 3 spacing 1.2\n"
            "set key box lt 1 lw 0.5\n",
            pipe);
    } else {
      fputs("unset key\n", pipe);
    }

    fputs("plot ", pipe);
    for (std::size_t i = 0; i < p.series.size(); ++i) {
      if (i) {
        fputs(", ", pipe);
      }
      const auto& e = p.series[i];
      if (e.style.find("lw") != std::string::npos
          || e.style.find("lc") != std::string::npos
          || e.style.find("ps") != std::string::npos) {
        fprintf(pipe, "$d_%d with %s", block_offset + (int)i, e.style.c_str());
      } else {
        fprintf(pipe,
                "$d_%d with %s ls %zu",
                block_offset + (int)i,
                e.style.c_str(),
                i + 1);
      }

      if (!e.label.empty()) {
        fprintf(pipe, " title '%s'", e.label.c_str());
      } else {
        fputs(" notitle", pipe);
      }
    }
    fputc('\n', pipe);
  }
}

inline void flush_to(FILE* pipe, const std::string& outfile) {
  auto& s = state();

  std::vector<Panel> all = s.panels;
  all.push_back(s.current);

  bool multiplot = (s.mp_rows_ > 0);

  // Terminal
  if (s.term_override_ == "dumb") {
    fprintf(pipe, "set terminal dumb size %d,%d\nset autoscale\n", s.term_w_, s.term_h_);
  } else if (outfile.empty()) {
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
  if (s.term_override_ != "dumb") {
    fputs("set style line 1 lt 1 lw 2 pt 7  ps 0.7 lc rgb '#2c3e50'\n", pipe);
    fputs("set style line 2 lt 2 lw 2 pt 5  ps 0.7 lc rgb '#c0392b'\n", pipe);
    fputs("set style line 3 lt 3 lw 2 pt 9  ps 0.7 lc rgb '#2980b9'\n", pipe);
    fputs("set style line 4 lt 4 lw 2 pt 13 ps 0.7 lc rgb '#27ae60'\n", pipe);
    fputs("set style line 5 lt 5 lw 2 pt 11 ps 0.7 lc rgb '#8e44ad'\n", pipe);
    fputs("set style line 100 lt 1 lw 0.5 lc rgb '#cccccc'\n", pipe);
    fputs("set grid back ls 100\n", pipe);
    fputs("set border 3 lw 1.5\n", pipe);
    fputs("set tics nomirror\n", pipe);
  }

  // Write datablocks
  int block = 0;
  for (const auto& p : all) {
    for (const auto& e : p.series) {
      fprintf(pipe, "$d_%d << EOD\n", block++);
      for (const auto& [x, y] : e.data) {
        fprintf(pipe, "%.15g %.15g\n", x, y);
      }
      fputs("EOD\n", pipe);
    }
    for (const auto& hm : p.heatmaps) {
      fprintf(pipe, "$d_%d << EOD\n", block++);
      for (int i = 0; i < hm.N; ++i) {
        double xi = (i + 1) * hm.h;
        for (int j = 0; j < hm.N; ++j) {
          fprintf(pipe,
                  "%.8g %.8g %.8g\n",
                  xi,
                  (j + 1) * hm.h,
                  hm.data[(static_cast<std::size_t>(i) * hm.N) + j]);
        }
        fputs("\n", pipe);
      }
      fputs("EOD\n", pipe);
    }
  }

  if (multiplot) {
    fprintf(pipe,
            "set multiplot layout %d,%d spacing 0.08,0.12\n",
            s.mp_rows_,
            s.mp_cols_);
    int off = 0;
    for (const auto& p : all) {
      write_panel(pipe, p, off);
      off += (int)p.series.size() + (int)p.heatmaps.size();
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

/// Append parallel x and y vectors to the current panel (supports std::vector,
/// num::Vector, etc.).
template<typename ContainerX, typename ContainerY>
inline void plot(const ContainerX& x,
                 const ContainerY& y,
                 const std::string& label = "",
                 const std::string& style = "lines") {
  Series s;
  s.reserve(x.size());
  for (std::size_t i = 0; i < x.size() && i < y.size(); ++i) {
    s.emplace_back(static_cast<double>(x[i]), static_cast<double>(y[i]));
  }
  detail::state().current.series.push_back({std::move(s), label, style});
}

// -- Decorators ---------------------------------------------------------------

/// Set the current panel title.
inline void title(const std::string& t) {
  detail::state().current.title_ = t;
}
/// Set the current panel x-axis label.
inline void xlabel(const std::string& l) {
  detail::state().current.xlabel_ = l;
}
/// Set the current panel y-axis label.
inline void ylabel(const std::string& l) {
  detail::state().current.ylabel_ = l;
}

/// Set the current panel x-axis limits.
inline void xlim(double lo, double hi) {
  detail::state().current.xrange_ =
    "[" + std::to_string(lo) + ":" + std::to_string(hi) + "]";
}
/// Set the current panel y-axis limits.
inline void ylim(double lo, double hi) {
  detail::state().current.yrange_ =
    "[" + std::to_string(lo) + ":" + std::to_string(hi) + "]";
}

/// Enable the legend for the current panel.
inline void legend() {
  detail::state().current.legend_ = true;
}

/// Plot component-wise path means and their min-max envelope.
template<class Paths, class Labels, class Colors>
inline void plot_paths(const Paths& paths,
                       const Labels& labels,
                       const Colors& colors,
                       const std::string& plot_title,
                       std::size_t samples = 1000) {
  if (paths.empty() || labels.empty() || labels.size() != colors.size() || samples < 2) {
    throw std::invalid_argument("invalid path plot data");
  }

  double first_end = std::numeric_limits<double>::infinity();
  double last_end = 0.0;
  for (const auto& path : paths) {
    if (path.times.empty() || path.states.empty()) {
      throw std::invalid_argument("path plot requires nonempty trajectories");
    }
    first_end = std::min(first_end, path.times.back());
    last_end = std::max(last_end, path.times.back());
  }

  std::vector<double> times(samples);
  for (std::size_t i = 0; i < samples; ++i) {
    times[i] = first_end * static_cast<double>(i) / static_cast<double>(samples - 1);
  }

  const std::size_t components = labels.size();
  std::vector<std::vector<double>> lower(
    components,
    std::vector<double>(samples, std::numeric_limits<double>::max()));
  std::vector<std::vector<double>> upper(
    components,
    std::vector<double>(samples, std::numeric_limits<double>::lowest()));
  std::vector<std::vector<double>> mean(components, std::vector<double>(samples, 0.0));

  for (const auto& path : paths) {
    std::size_t state = 0;
    for (std::size_t i = 0; i < samples; ++i) {
      while (state + 1 < path.times.size() && path.times[state + 1] <= times[i]) {
        ++state;
      }
      for (std::size_t component = 0; component < components; ++component) {
        const double value = path.states[state][component];
        lower[component][i] = std::min(lower[component][i], value);
        upper[component][i] = std::max(upper[component][i], value);
        mean[component][i] += value / static_cast<double>(paths.size());
      }
    }
  }

  if (paths.size() > 1) {
    for (std::size_t component = 0; component < components; ++component) {
      Series envelope;
      envelope.reserve(2 * samples);
      for (std::size_t i = 0; i < samples; ++i) {
        envelope.emplace_back(times[i], upper[component][i]);
      }
      for (std::size_t i = samples; i-- > 0;) {
        envelope.emplace_back(times[i], lower[component][i]);
      }
      plot(envelope,
           std::string{},
           "filledcurves closed fs transparent solid 0.18 noborder lc rgb '"
             + std::string(colors[component]) + "'");
    }
  }

  for (std::size_t component = 0; component < components; ++component) {
    plot(times,
         mean[component],
         std::string(labels[component]),
         "lines lw 2 lc rgb '" + std::string(colors[component]) + "'");
  }

  std::ostringstream endpoint;
  endpoint << std::setprecision(3) << "  endpoint=[" << first_end << ", " << last_end
           << "]";
  title(plot_title + endpoint.str());
  xlabel("t");
  ylabel("count");
  legend();
}

/// Use logarithmic x and y axes on the current panel.
inline void loglog() {
  detail::state().current.logx_ = detail::state().current.logy_ = true;
}
/// Use a logarithmic y axis on the current panel.
inline void semilogy() {
  detail::state().current.logy_ = true;
}
/// Use a logarithmic x axis on the current panel.
inline void semilogx() {
  detail::state().current.logx_ = true;
}

// -- Multiplot ----------------------------------------------------------------

/// @brief Start a multiplot with the given grid dimensions.
inline void subplot(int rows, int cols = 1) {
  detail::state().reset();
  detail::state().mp_rows_ = rows;
  detail::state().mp_cols_ = cols;
}

/// @brief Advance to the next panel.
inline void next() {
  detail::state().panels.push_back(detail::state().current);
  detail::state().current = detail::Panel{};
}

// -- 2-D heatmap --------------------------------------------------------------

template<typename Container>
/// Add an N-by-N row-major scalar array as a heatmap.
inline void heatmap(const Container& u,
                    int N,
                    double h,
                    double vmin = 0.0,
                    double vmax = 1.0) {
  detail::HeatmapEntry e;
  e.data.assign(u.data(), u.data() + u.size());
  e.N = N;
  e.h = h;
  e.vmin = vmin;
  e.vmax = vmax;
  detail::state().current.heatmaps.push_back(std::move(e));
}

template<class Field>
/// Add a square field exposing vec(), N(), and h() as a heatmap.
inline void heatmap(const Field& g, double vmin = 0.0, double vmax = 1.0) {
  heatmap(g.vec(), g.N(), g.h(), vmin, vmax);
}

/// Set a gnuplot palette expression for the current panel.
inline void colormap(const std::string& palette) {
  detail::state().current.palette_ = palette;
}

// -- In-Terminal ASCII Plotting -----------------------------------------------

/// Configure terminal ASCII mode with custom width and height.
inline void terminal_dumb(int width = 140, int height = 35) {
  detail::state().term_override_ = "dumb";
  detail::state().term_w_ = width;
  detail::state().term_h_ = height;
}

/// Render ASCII plot directly to stdout/terminal window.
inline void show_dumb(int width = 140, int height = 35) {
  terminal_dumb(width, height);
  FILE* pipe = popen("gnuplot", "w");
  if (!pipe) {
    throw std::runtime_error("could not open gnuplot -- is it installed?");
  }
  detail::flush_to(pipe, "");
  fflush(pipe);
  pclose(pipe);
  detail::state().reset();
}

// -- Output -------------------------------------------------------------------

/// Render accumulated panels in an interactive gnuplot window and clear state.
inline void show() {
  FILE* pipe = popen("gnuplot", "w");
  if (!pipe) {
    throw std::runtime_error("could not open gnuplot -- is it installed?");
  }
  detail::flush_to(pipe, "");
  if (detail::state().term_override_ != "dumb") {
    fputs("pause mouse close\n", pipe);
  }
  fflush(pipe);
  pclose(pipe);
  detail::state().reset();
}

/// Render accumulated panels to a file selected by its extension and clear state.
inline void savefig(const std::string& filename) {
  FILE* pipe = popen("gnuplot", "w");
  if (!pipe) {
    throw std::runtime_error("could not open gnuplot -- is it installed?");
  }
  detail::flush_to(pipe, filename);
  fflush(pipe);
  pclose(pipe);
  detail::state().reset();
}

/// Clear all accumulated plot state without rendering.
inline void clf() {
  detail::state().reset();
}

} // namespace num::plt
