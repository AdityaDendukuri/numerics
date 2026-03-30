/// @file bench_plot.hpp
/// @brief SIAM-style benchmark plots built on num::Gnuplot (plot/plot.hpp).
///
/// Usage from a custom benchmark main:
///
///   CollectingReporter rep;
///   benchmark::RunSpecifiedBenchmarks(&rep);
///   if (do_plot)
///       bench_plot::plot_all(rep.results(), "plots");
#pragma once

#include <benchmark/benchmark.h>
#include "plot/plot.hpp"

#include <map>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

namespace bench_plot {

using Run    = benchmark::BenchmarkReporter::Run;
using num::Point;
using num::Series;
using num::Gnuplot;
using num::apply_siam_style;
using num::set_loglog;
using num::set_logx;

// Result helpers

/// Returns true if the benchmark name contains all of the given substrings.
static bool name_has(const Run& r, std::initializer_list<const char*> needles) {
    const auto& n = r.benchmark_name();
    for (auto s : needles)
        if (n.find(s) == std::string::npos) return false;
    return true;
}

/// Returns true if the benchmark name contains none of the given substrings.
static bool name_lacks(const Run& r, std::initializer_list<const char*> needles) {
    const auto& n = r.benchmark_name();
    for (auto s : needles)
        if (n.find(s) != std::string::npos) return false;
    return true;
}

/// Extract the numeric size argument.
/// Prefers complexity_n (set by SetComplexityN); falls back to parsing /N from the name.
static double size_of(const Run& r) {
    if (r.complexity_n > 0)
        return static_cast<double>(r.complexity_n);
    const auto& name = r.benchmark_name();
    auto pos = name.rfind('/');
    if (pos != std::string::npos)
        try { return std::stod(name.substr(pos + 1)); } catch (...) {}
    return 0.0;
}

/// Real time in microseconds per iteration.
static double time_us(const Run& r) {
    return r.GetAdjustedRealTime() / 1e3;
}

/// Counter value, 0 if not present.
static double counter(const Run& r, const std::string& key) {
    auto it = r.counters.find(key);
    return it != r.counters.end() ? double(it->second) : 0.0;
}

/// Build a (size, value) series from runs matching a name substring.
static Series series_by_counter(const std::vector<Run>& runs,
                                 const std::string& name_substr,
                                 const std::string& counter_key) {
    Series s;
    for (auto& r : runs) {
        if (r.benchmark_name().find(name_substr) == std::string::npos) continue;
        if (r.benchmark_name().find("BigO") != std::string::npos) continue;
        if (r.benchmark_name().find("RMS")  != std::string::npos) continue;
        double v = counter(r, counter_key);
        if (v > 0) s.emplace_back(size_of(r), v);
    }
    std::sort(s.begin(), s.end());
    return s;
}

static Series series_by_time(const std::vector<Run>& runs,
                              const std::string& name_substr) {
    Series s;
    for (auto& r : runs) {
        if (r.benchmark_name().find(name_substr) == std::string::npos) continue;
        if (r.benchmark_name().find("BigO") != std::string::npos) continue;
        if (r.benchmark_name().find("RMS")  != std::string::npos) continue;
        s.emplace_back(size_of(r), time_us(r));
    }
    std::sort(s.begin(), s.end());
    return s;
}

// Build a plot command string for N inline series.
// series_labels: pairs of (label, ls_index)
static std::string plot_cmd(const std::vector<std::pair<std::string,int>>& series) {
    std::string cmd = "plot ";
    for (size_t i = 0; i < series.size(); ++i) {
        if (i) cmd += ", ";
        cmd += "'-' with linespoints ls " + std::to_string(series[i].second)
             + " title '" + series[i].first + "'";
    }
    cmd += "\n";
    return cmd;
}

// -- Individual plot functions -------------------------------------------------

/// matmul.pdf  -- GFLOP/s vs n for every matmul variant
static void plot_matmul(Gnuplot& gp, const std::vector<Run>& runs,
                         const std::string& outdir,
                         const std::string& ext = ".pdf") {
    struct Variant { std::string key; std::string label; int ls; };
    std::vector<Variant> variants = {
        {"Matmul_Naive",      "naive",       1},
        {"Matmul_Blocked",    "blocked",     2},
        {"Matmul_RegBlocked", "reg-blocked", 3},
        {"Backend::simd",     "simd",        4},
        {"Backend::blas",     "blas",        5},
        {"Backend::omp",      "omp",         6},
    };

    // collect non-empty series first
    std::vector<std::pair<Variant, Series>> data;
    for (auto& v : variants) {
        auto s = series_by_counter(runs, v.key, "GFLOP/s");
        if (!s.empty()) data.emplace_back(v, std::move(s));
    }
    if (data.empty()) return;

    gp << "set output '" + outdir + "/matmul" + ext + "'\n"
       << "set title 'Matrix Multiply: policy comparison' font 'Times-Bold,14'\n"
       << "set xlabel 'n' font 'Times,12'\n"
       << "set ylabel 'GFLOP/s' font 'Times,12'\n";
    set_loglog(gp);

    std::vector<std::pair<std::string,int>> labels;
    for (auto& [v, _] : data) labels.emplace_back(v.label, v.ls);
    gp << plot_cmd(labels);
    for (auto& [_, s] : data) gp.send1d(s);
}

/// matvec.pdf  -- GB/s vs n for every matvec policy
static void plot_matvec(Gnuplot& gp, const std::vector<Run>& runs,
                         const std::string& outdir,
                         const std::string& ext = ".pdf") {
    struct Variant { std::string key; std::string label; int ls; };
    std::vector<Variant> variants = {
        {"BM_Matvec<num::Backend::seq",     "seq",     1},
        {"BM_Matvec<num::Backend::blocked", "blocked", 2},
        {"BM_Matvec<num::Backend::simd",    "simd",    3},
        {"BM_Matvec<num::Backend::blas",    "blas",    4},
        {"BM_Matvec<num::Backend::omp",     "omp",     5},
    };

    std::vector<std::pair<Variant, Series>> data;
    for (auto& v : variants) {
        Series s;
        for (auto& r : runs) {
            if (r.benchmark_name().find(v.key) == std::string::npos) continue;
            double bps = counter(r, "bytes_per_second");
            if (bps > 0) s.emplace_back(size_of(r), bps / 1e9);
        }
        std::sort(s.begin(), s.end());
        if (!s.empty()) data.emplace_back(v, std::move(s));
    }
    if (data.empty()) return;

    gp << "set output '" + outdir + "/matvec" + ext + "'\n"
       << "set title 'Matrix-Vector Multiply: memory bandwidth' font 'Times-Bold,14'\n"
       << "set xlabel 'n' font 'Times,12'\n"
       << "set ylabel 'GB/s' font 'Times,12'\n";
    set_loglog(gp);

    std::vector<std::pair<std::string,int>> labels;
    for (auto& [v, _] : data) labels.emplace_back(v.label, v.ls);
    gp << plot_cmd(labels);
    for (auto& [_, s] : data) gp.send1d(s);
}

/// dot.pdf  -- GB/s vs n for dot/axpy
static void plot_dot_axpy(Gnuplot& gp, const std::vector<Run>& runs,
                           const std::string& outdir,
                           const std::string& ext = ".pdf") {
    struct Op { std::string op_key; std::string op_label; };
    for (auto& op : std::vector<Op>{{"Dot", "dot"}, {"Axpy", "axpy"}}) {
        struct V { std::string key; std::string label; int ls; };
        std::vector<V> variants = {
            {"BM_" + op.op_key + "<num::Backend::seq",  "seq",  1},
            {"BM_" + op.op_key + "<num::Backend::blas", "blas", 2},
            {"BM_" + op.op_key + "<num::Backend::omp",  "omp",  3},
        };

        std::vector<std::pair<V, Series>> data;
        for (auto& v : variants) {
            Series s;
            for (auto& r : runs) {
                if (r.benchmark_name().find(v.key) == std::string::npos) continue;
                double bps = counter(r, "bytes_per_second");
                if (bps > 0) s.emplace_back(size_of(r), bps / 1e9);
            }
            std::sort(s.begin(), s.end());
            if (!s.empty()) data.emplace_back(v, std::move(s));
        }
        if (data.empty()) continue;

        gp << "set output '" + outdir + "/" + op.op_label + ext + "'\n"
           << "set title '" + op.op_label + ": memory bandwidth' font 'Times-Bold,14'\n"
           << "set xlabel 'n' font 'Times,12'\n"
           << "set ylabel 'GB/s' font 'Times,12'\n";
        set_loglog(gp);

        std::vector<std::pair<std::string,int>> labels;
        for (auto& [v, _] : data) labels.emplace_back(v.label, v.ls);
        gp << plot_cmd(labels);
        for (auto& [_, s] : data) gp.send1d(s);
    }
}

/// cg.pdf  -- time (mus) vs n for CG solver
static void plot_cg(Gnuplot& gp, const std::vector<Run>& runs,
                     const std::string& outdir,
                     const std::string& ext = ".pdf") {
    auto cpu = series_by_time(runs, "BM_CG/");
    auto gpu = series_by_time(runs, "BM_CG_GPU");
    if (cpu.empty() && gpu.empty()) return;

    gp << "set output '" + outdir + "/cg" + ext + "'\n"
       << "set title 'Conjugate Gradient: time vs system size' font 'Times-Bold,14'\n"
       << "set xlabel 'n' font 'Times,12'\n"
       << "set ylabel 'Time ({/Symbol m}s)' font 'Times,12'\n";
    set_loglog(gp);

    if (!cpu.empty() && !gpu.empty()) {
        gp << plot_cmd({{"CPU", 1}, {"GPU", 2}});
        gp.send1d(cpu);
        gp.send1d(gpu);
    } else if (!cpu.empty()) {
        gp << plot_cmd({{"CG", 1}});
        gp.send1d(cpu);
    } else {
        gp << plot_cmd({{"GPU", 2}});
        gp.send1d(gpu);
    }
}

/// thomas.pdf  -- time (mus) vs n for Thomas algorithm
static void plot_thomas(Gnuplot& gp, const std::vector<Run>& runs,
                         const std::string& outdir,
                         const std::string& ext = ".pdf") {
    auto data = series_by_time(runs, "BM_Thomas/");
    if (data.empty()) return;

    gp << "set output '" + outdir + "/thomas" + ext + "'\n"
       << "set title 'Thomas Algorithm: O(n) scaling' font 'Times-Bold,14'\n"
       << "set xlabel 'n' font 'Times,12'\n"
       << "set ylabel 'Time ({/Symbol m}s)' font 'Times,12'\n";
    set_loglog(gp);
    gp << plot_cmd({{"Thomas", 1}});
    gp.send1d(data);
}

/// fft.pdf  -- GB/s vs n: seq vs fftw for one-shot and plan FFT
static void plot_fft(Gnuplot& gp, const std::vector<Run>& runs,
                     const std::string& outdir,
                     const std::string& ext = ".pdf") {
    struct Variant { std::string key; std::string label; int ls; };

    // -- forward FFT: one-shot -----------------------------------------------
    {
        std::vector<Variant> variants = {
            {"BM_FFT<FFTBackend::seq",     "seq (Cooley-Tukey)",  1},
            {"BM_FFT<FFTBackend::simd",    "simd (AVX2/NEON)",    2},
            {"BM_FFT<FFTBackend::stdsimd", "std::simd",           3},
            {"BM_FFT<FFTBackend::fftw",    "fftw (FFTW3)",        4},
        };
        std::vector<std::pair<Variant, Series>> data;
        for (auto& v : variants) {
            Series s;
            for (auto& r : runs) {
                if (r.benchmark_name().find(v.key) == std::string::npos) continue;
                if (r.benchmark_name().find("BigO") != std::string::npos) continue;
                if (r.benchmark_name().find("RMS")  != std::string::npos) continue;
                double bps = counter(r, "bytes_per_second");
                if (bps > 0) s.emplace_back(size_of(r), bps / 1e9);
            }
            std::sort(s.begin(), s.end());
            if (!s.empty()) data.emplace_back(v, std::move(s));
        }
        if (!data.empty()) {
            gp << "set output '" + outdir + "/fft" + ext + "'\n"
               << "set title 'FFT (forward): memory throughput' font 'Times-Bold,14'\n"
               << "set xlabel 'n' font 'Times,12'\n"
               << "set ylabel 'GB/s' font 'Times,12'\n";
            set_loglog(gp);
            std::vector<std::pair<std::string,int>> labels;
            for (auto& [v, _] : data) labels.emplace_back(v.label, v.ls);
            gp << plot_cmd(labels);
            for (auto& [_, s] : data) gp.send1d(s);
        }
    }

    // -- reusable plan: seq vs fftw ------------------------------------------
    {
        std::vector<Variant> variants = {
            {"BM_FFTPlan<FFTBackend::seq",     "seq plan (Cooley-Tukey)",  1},
            {"BM_FFTPlan<FFTBackend::simd",    "simd plan (AVX2/NEON)",    2},
            {"BM_FFTPlan<FFTBackend::stdsimd", "std::simd plan",           3},
            {"BM_FFTPlan<FFTBackend::fftw",    "fftw plan (FFTW3)",        4},
        };
        std::vector<std::pair<Variant, Series>> data;
        for (auto& v : variants) {
            Series s;
            for (auto& r : runs) {
                if (r.benchmark_name().find(v.key) == std::string::npos) continue;
                if (r.benchmark_name().find("BigO") != std::string::npos) continue;
                if (r.benchmark_name().find("RMS")  != std::string::npos) continue;
                double bps = counter(r, "bytes_per_second");
                if (bps > 0) s.emplace_back(size_of(r), bps / 1e9);
            }
            std::sort(s.begin(), s.end());
            if (!s.empty()) data.emplace_back(v, std::move(s));
        }
        if (!data.empty()) {
            gp << "set output '" + outdir + "/fft_plan" + ext + "'\n"
               << "set title 'FFT Plan (reusable): memory throughput' font 'Times-Bold,14'\n"
               << "set xlabel 'n' font 'Times,12'\n"
               << "set ylabel 'GB/s' font 'Times,12'\n";
            set_loglog(gp);
            std::vector<std::pair<std::string,int>> labels;
            for (auto& [v, _] : data) labels.emplace_back(v.label, v.ls);
            gp << plot_cmd(labels);
            for (auto& [_, s] : data) gp.send1d(s);
        }
    }

    // -- rfft: seq vs fftw ---------------------------------------------------
    {
        std::vector<Variant> variants = {
            {"BM_RFFT<FFTBackend::seq",     "seq (Cooley-Tukey)",  1},
            {"BM_RFFT<FFTBackend::simd",    "simd (AVX2/NEON)",    2},
            {"BM_RFFT<FFTBackend::stdsimd", "std::simd",           3},
            {"BM_RFFT<FFTBackend::fftw",    "fftw (FFTW3)",        4},
        };
        std::vector<std::pair<Variant, Series>> data;
        for (auto& v : variants) {
            Series s;
            for (auto& r : runs) {
                if (r.benchmark_name().find(v.key) == std::string::npos) continue;
                if (r.benchmark_name().find("BigO") != std::string::npos) continue;
                if (r.benchmark_name().find("RMS")  != std::string::npos) continue;
                double bps = counter(r, "bytes_per_second");
                if (bps > 0) s.emplace_back(size_of(r), bps / 1e9);
            }
            std::sort(s.begin(), s.end());
            if (!s.empty()) data.emplace_back(v, std::move(s));
        }
        if (!data.empty()) {
            gp << "set output '" + outdir + "/rfft" + ext + "'\n"
               << "set title 'rFFT (real-to-complex): memory throughput' font 'Times-Bold,14'\n"
               << "set xlabel 'n' font 'Times,12'\n"
               << "set ylabel 'GB/s' font 'Times,12'\n";
            set_loglog(gp);
            std::vector<std::pair<std::string,int>> labels;
            for (auto& [v, _] : data) labels.emplace_back(v.label, v.ls);
            gp << plot_cmd(labels);
            for (auto& [_, s] : data) gp.send1d(s);
        }
    }
}

/// banded.pdf  -- time (mus) vs n for banded solver variants
static void plot_banded(Gnuplot& gp, const std::vector<Run>& runs,
                         const std::string& outdir,
                         const std::string& ext = ".pdf") {
    auto data = series_by_time(runs, "BM_Band");
    if (data.empty()) return;
    gp << "set output '" + outdir + "/banded" + ext + "'\n"
       << "set title 'Banded Solver: time vs system size' font 'Times-Bold,14'\n"
       << "set xlabel 'n' font 'Times,12'\n"
       << "set ylabel 'Time ({/Symbol m}s)' font 'Times,12'\n";
    set_loglog(gp);
    gp << plot_cmd({{"banded", 1}});
    gp.send1d(data);
}

/// lu.pdf  -- GFLOP/s vs n: our seq vs our omp vs LAPACK
static void plot_lu(Gnuplot& gp, const std::vector<Run>& runs,
                    const std::string& outdir,
                    const std::string& ext = ".pdf") {
    struct Variant { std::string key; std::string label; int ls; };
    std::vector<Variant> variants = {
        {"BM_LU<num::Backend::seq",    "our (seq)",    1},
        {"BM_LU<num::Backend::omp",    "our (omp)",    2},
        {"BM_LU<num::Backend::lapack", "LAPACK dgetrf",3},
    };
    std::vector<std::pair<Variant, Series>> data;
    for (auto& v : variants) {
        auto s = series_by_counter(runs, v.key, "GFLOP/s");
        if (!s.empty()) data.emplace_back(v, std::move(s));
    }
    if (data.empty()) return;
    gp << "set output '" + outdir + "/lu" + ext + "'\n"
       << "set title 'LU Factorization: GFLOP/s vs n' font 'Times-Bold,14'\n"
       << "set xlabel 'n' font 'Times,12'\n"
       << "set ylabel 'GFLOP/s' font 'Times,12'\n";
    set_loglog(gp);
    std::vector<std::pair<std::string,int>> labels;
    for (auto& [v, _] : data) labels.emplace_back(v.label, v.ls);
    gp << plot_cmd(labels);
    for (auto& [_, s] : data) gp.send1d(s);
}

/// qr.pdf  -- GFLOP/s vs n: our seq vs our omp vs LAPACK
static void plot_qr(Gnuplot& gp, const std::vector<Run>& runs,
                    const std::string& outdir,
                    const std::string& ext = ".pdf") {
    struct Variant { std::string key; std::string label; int ls; };
    std::vector<Variant> variants = {
        {"BM_QR<num::Backend::seq",    "our (seq)",    1},
        {"BM_QR<num::Backend::omp",    "our (omp)",    2},
        {"BM_QR<num::Backend::lapack", "LAPACK dgeqrf",3},
    };
    std::vector<std::pair<Variant, Series>> data;
    for (auto& v : variants) {
        auto s = series_by_counter(runs, v.key, "GFLOP/s");
        if (!s.empty()) data.emplace_back(v, std::move(s));
    }
    if (data.empty()) return;
    gp << "set output '" + outdir + "/qr" + ext + "'\n"
       << "set title 'QR Factorization: GFLOP/s vs n' font 'Times-Bold,14'\n"
       << "set xlabel 'n' font 'Times,12'\n"
       << "set ylabel 'GFLOP/s' font 'Times,12'\n";
    set_loglog(gp);
    std::vector<std::pair<std::string,int>> labels;
    for (auto& [v, _] : data) labels.emplace_back(v.label, v.ls);
    gp << plot_cmd(labels);
    for (auto& [_, s] : data) gp.send1d(s);
}

/// svd.pdf  -- time (mus) vs n: Jacobi vs randomized vs LAPACK
static void plot_svd(Gnuplot& gp, const std::vector<Run>& runs,
                     const std::string& outdir,
                     const std::string& ext = ".pdf") {
    struct Variant { std::string key; std::string label; int ls; };
    std::vector<Variant> variants = {
        {"BM_SVD<num::Backend::seq",    "our Jacobi",      1},
        {"BM_SVD_Randomized",           "randomized (k=n/8)", 2},
        {"BM_SVD<num::Backend::lapack", "LAPACK dgesdd",   3},
    };
    std::vector<std::pair<Variant, Series>> data;
    for (auto& v : variants) {
        auto s = series_by_counter(runs, v.key, "GFLOP/s");
        if (s.empty()) s = series_by_time(runs, v.key);   // fallback to time if no GFLOP/s counter
        if (!s.empty()) data.emplace_back(v, std::move(s));
    }
    if (data.empty()) return;
    gp << "set output '" + outdir + "/svd" + ext + "'\n"
       << "set title 'SVD: time ({/Symbol m}s) vs n' font 'Times-Bold,14'\n"
       << "set xlabel 'n' font 'Times,12'\n"
       << "set ylabel 'Time ({/Symbol m}s)' font 'Times,12'\n";
    set_loglog(gp);
    std::vector<std::pair<std::string,int>> labels;
    for (auto& [v, _] : data) labels.emplace_back(v.label, v.ls);
    gp << plot_cmd(labels);
    for (auto& [_, s] : data) gp.send1d(s);
}

/// eigen.pdf  -- time (mus) vs n: Jacobi vs Lanczos vs LAPACK
static void plot_eigen(Gnuplot& gp, const std::vector<Run>& runs,
                       const std::string& outdir,
                       const std::string& ext = ".pdf") {
    struct Variant { std::string key; std::string label; int ls; };
    std::vector<Variant> variants = {
        {"BM_EigSym<num::Backend::seq",    "our Jacobi",       1},
        {"BM_Lanczos",                     "Lanczos (k=10)",   2},
        {"BM_EigSym<num::Backend::lapack", "LAPACK dsyevd",    3},
    };
    std::vector<std::pair<Variant, Series>> data;
    for (auto& v : variants) {
        auto s = series_by_time(runs, v.key);
        if (!s.empty()) data.emplace_back(v, std::move(s));
    }
    if (data.empty()) return;
    gp << "set output '" + outdir + "/eigen" + ext + "'\n"
       << "set title 'Symmetric Eigensolver: time ({/Symbol m}s) vs n' font 'Times-Bold,14'\n"
       << "set xlabel 'n' font 'Times,12'\n"
       << "set ylabel 'Time ({/Symbol m}s)' font 'Times,12'\n";
    set_loglog(gp);
    std::vector<std::pair<std::string,int>> labels;
    for (auto& [v, _] : data) labels.emplace_back(v.label, v.ls);
    gp << plot_cmd(labels);
    for (auto& [_, s] : data) gp.send1d(s);
}

// -- Entry point ---------------------------------------------------------------

/// Generate all plots for the collected benchmark results.
/// @param runs    Results from CollectingReporter.
/// @param outdir  Directory to write PDF files into (must exist).
inline void plot_all(const std::vector<Run>& runs,
                     const std::string& outdir = "plots") {
    Gnuplot gp;
    gp << "set terminal pdfcairo enhanced font 'Times,12' size 4in,3in linewidth 1.5\n";
    apply_siam_style(gp);

    plot_matmul(gp,    runs, outdir);
    plot_matvec(gp,    runs, outdir);
    plot_dot_axpy(gp,  runs, outdir);
    plot_cg(gp,        runs, outdir);
    plot_thomas(gp,    runs, outdir);
    plot_fft(gp,       runs, outdir);
    plot_banded(gp,    runs, outdir);
    plot_lu(gp,        runs, outdir);
    plot_qr(gp,        runs, outdir);
    plot_svd(gp,       runs, outdir);
    plot_eigen(gp,     runs, outdir);
}

/// Same as plot_all but writes PNGs for report embedding.
inline void plot_all_png(const std::vector<Run>& runs,
                          const std::string& outdir = "plots") {
    Gnuplot gp;
    gp << "set terminal pngcairo enhanced font 'Liberation Sans,11' size 800,600\n";
    apply_siam_style(gp);
    plot_matmul(gp,    runs, outdir, ".png");
    plot_matvec(gp,    runs, outdir, ".png");
    plot_dot_axpy(gp,  runs, outdir, ".png");
    plot_cg(gp,        runs, outdir, ".png");
    plot_thomas(gp,    runs, outdir, ".png");
    plot_fft(gp,       runs, outdir, ".png");
    plot_banded(gp,    runs, outdir, ".png");
    plot_lu(gp,        runs, outdir, ".png");
    plot_qr(gp,        runs, outdir, ".png");
    plot_svd(gp,       runs, outdir, ".png");
    plot_eigen(gp,     runs, outdir, ".png");
}

// ─── ASCII / dumb-terminal plots ─────────────────────────────────────────────
//
// GnuplotAscii builds a gnuplot script and runs it with "set terminal dumb",
// writing ASCII art to a .txt file.  gen_report then reads those files and
// wraps them in ``` code blocks -- works in GitHub Actions job summary, local
// terminals, and any plain-text viewer.

/// Thin wrapper around a gnuplot pipe for ASCII (dumb terminal) output.
class GnuplotAscii {
    std::string script_;
public:
    GnuplotAscii& operator<<(const std::string& cmd) { script_ += cmd; return *this; }

    void send1d(const Series& data) {
        for (auto& [x, y] : data)
            script_ += std::to_string(x) + " " + std::to_string(y) + "\n";
        script_ += "e\n";
    }

    /// Run gnuplot with dumb terminal; write ASCII art to outpath.
    void write(const std::string& outpath, int w = 78, int h = 18) const {
        FILE* pipe = popen("gnuplot 2>/dev/null", "w");
        if (!pipe) return;
        std::string hdr =
            "set terminal dumb " + std::to_string(w) + " " + std::to_string(h) + " noenhanced\n"
            + "set output '" + outpath + "'\n";
        fputs(hdr.c_str(), pipe);
        fputs(script_.c_str(), pipe);
        pclose(pipe);
    }
};

// Helper: collect a GB/s series for all named variants, write one ASCII plot.
static void ascii_throughput(
    const std::vector<Run>& runs,
    const std::string& outpath,
    const std::string& title,
    const std::string& ylabel,
    const std::vector<std::pair<std::string,std::string>>& variants, // key, label
    bool loglog = true)
{
    std::vector<std::pair<std::string, Series>> data;
    for (auto& [key, label] : variants) {
        Series s;
        for (auto& r : runs) {
            if (r.benchmark_name().find(key)   == std::string::npos) continue;
            if (r.benchmark_name().find("BigO") != std::string::npos) continue;
            if (r.benchmark_name().find("RMS")  != std::string::npos) continue;
            double bps = counter(r, "bytes_per_second");
            if (bps > 0) s.emplace_back(size_of(r), bps / 1e9);
        }
        std::sort(s.begin(), s.end());
        if (!s.empty()) data.emplace_back(label, std::move(s));
    }
    if (data.empty()) return;

    GnuplotAscii gp;
    gp << "set title '" + title + "'\n"
       << "set xlabel 'n'\n"
       << "set ylabel '" + ylabel + "'\n"
       << "set grid\n"
       << "set key top left\n";
    if (loglog) gp << "set logscale xy\n";

    std::string cmd = "plot ";
    for (size_t i = 0; i < data.size(); ++i) {
        if (i) cmd += ", ";
        cmd += "'-' with linespoints title '" + data[i].first + "'";
    }
    gp << cmd + "\n";
    for (auto& [_, s] : data) gp.send1d(s);
    gp.write(outpath);
}

static void ascii_time(
    const std::vector<Run>& runs,
    const std::string& outpath,
    const std::string& title,
    const std::vector<std::pair<std::string,std::string>>& variants)
{
    std::vector<std::pair<std::string, Series>> data;
    for (auto& [key, label] : variants) {
        Series s;
        for (auto& r : runs) {
            if (r.benchmark_name().find(key)   == std::string::npos) continue;
            if (r.benchmark_name().find("BigO") != std::string::npos) continue;
            if (r.benchmark_name().find("RMS")  != std::string::npos) continue;
            s.emplace_back(size_of(r), time_us(r));
        }
        std::sort(s.begin(), s.end());
        if (!s.empty()) data.emplace_back(label, std::move(s));
    }
    if (data.empty()) return;

    GnuplotAscii gp;
    gp << "set title '" + title + "'\n"
       << "set xlabel 'n'\n"
       << "set ylabel 'us'\n"
       << "set grid\n"
       << "set key top left\n"
       << "set logscale xy\n";

    std::string cmd = "plot ";
    for (size_t i = 0; i < data.size(); ++i) {
        if (i) cmd += ", ";
        cmd += "'-' with linespoints title '" + data[i].first + "'";
    }
    gp << cmd + "\n";
    for (auto& [_, s] : data) gp.send1d(s);
    gp.write(outpath);
}

/// Generate all ASCII plots (.txt files) into outdir.
inline void plot_all_ascii(const std::vector<Run>& runs,
                            const std::string& outdir = "output/plots")
{
    // matmul (GFLOP/s via counter)
    {
        std::vector<std::pair<std::string,std::string>> v = {
            {"Matmul_Naive",      "naive"},
            {"Matmul_Blocked",    "blocked"},
            {"Matmul_RegBlocked", "reg-blocked"},
            {"Backend::simd",     "simd"},
            {"Backend::blas",     "blas"},
            {"Backend::omp",      "omp"},
        };
        std::vector<std::pair<std::string, Series>> data;
        for (auto& [key, label] : v) {
            auto s = series_by_counter(runs, key, "GFLOP/s");
            if (!s.empty()) data.emplace_back(label, std::move(s));
        }
        if (!data.empty()) {
            GnuplotAscii gp;
            gp << "set title 'Matrix Multiply: GFLOP/s vs n'\n"
               << "set xlabel 'n'\nset ylabel 'GFLOP/s'\nset grid\n"
               << "set key top left\nset logscale xy\n";
            std::string cmd = "plot ";
            for (size_t i = 0; i < data.size(); ++i) {
                if (i) cmd += ", ";
                cmd += "'-' with linespoints title '" + data[i].first + "'";
            }
            gp << cmd + "\n";
            for (auto& [_, s] : data) gp.send1d(s);
            gp.write(outdir + "/matmul.txt");
        }
    }

    ascii_throughput(runs, outdir + "/matvec.txt",
        "Matrix-Vector Multiply: GB/s vs n", "GB/s", {
            {"BM_Matvec<num::Backend::seq",     "seq"},
            {"BM_Matvec<num::Backend::blocked",  "blocked"},
            {"BM_Matvec<num::Backend::simd",     "simd"},
            {"BM_Matvec<num::Backend::blas",     "blas"},
            {"BM_Matvec<num::Backend::omp",      "omp"},
        });

    for (auto& op : std::vector<std::pair<std::string,std::string>>{{"Dot","dot"},{"Axpy","axpy"}})
        ascii_throughput(runs, outdir + "/" + op.second + ".txt",
            op.second + ": GB/s vs n", "GB/s", {
                {"BM_" + op.first + "<num::Backend::seq",  "seq"},
                {"BM_" + op.first + "<num::Backend::blas", "blas"},
                {"BM_" + op.first + "<num::Backend::omp",  "omp"},
            });

    ascii_time(runs, outdir + "/cg.txt",     "Conjugate Gradient: time vs n",  {{"BM_CG/", "CG"}});
    ascii_time(runs, outdir + "/thomas.txt", "Thomas Algorithm: time vs n",     {{"BM_Thomas/", "Thomas"}});
    ascii_time(runs, outdir + "/banded.txt", "Banded Solver: time vs n",        {{"BM_Band", "banded"}});

    // FFT: forward, plan, rfft -- one file each
    for (auto& [bm, file, title] : std::vector<std::tuple<std::string,std::string,std::string>>{
            {"BM_FFT<",     "fft.txt",      "FFT (forward): GB/s vs n"},
            {"BM_FFTPlan<", "fft_plan.txt", "FFT Plan (reusable): GB/s vs n"},
            {"BM_RFFT<",    "rfft.txt",     "rFFT (real-to-complex): GB/s vs n"},
        })
    {
        ascii_throughput(runs, outdir + "/" + file, title, "GB/s", {
            {bm + "FFTBackend::seq",     "seq"},
            {bm + "FFTBackend::simd",    "simd"},
            {bm + "FFTBackend::stdsimd", "std::simd"},
            {bm + "FFTBackend::fftw",    "fftw"},
        });
    }
}

} // namespace bench_plot
