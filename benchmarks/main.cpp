/// @file benchmarks/main.cpp
/// @brief Custom benchmark runner with optional gnuplot output and memory reporting.
///
/// Drop-in replacement for benchmark::benchmark_main.
/// Adds custom flags on top of the standard Google Benchmark flags:
///
///   --plot[=DIR]      Run benchmarks, then write SIAM-style PDFs to DIR
///                     (default: output/plots/).  Requires gnuplot in PATH.
///   --report[=DIR]    Like --plot but writes PNG instead of PDF.
///   --memreport       After benchmarks finish, print a heap-allocation
///                     comparison table (allocs/iter, bytes/iter, peak).
///
/// All other --benchmark_* flags are forwarded to Google Benchmark unchanged.

#include <benchmark/benchmark.h>
#include "bench_memory.hpp"
#include "bench_plot.hpp"

#include <cstring>
#include <deque>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

/// Collects benchmark results while still printing them to the console.
///
/// `Run::memory_result` is a raw pointer into the benchmark runner's
/// internal storage, which is freed when RunSpecifiedBenchmarks returns.
/// We deep-copy each Result so that print_report() can safely access it
/// after the runner has finished.
class CollectingReporter : public benchmark::ConsoleReporter {
public:
    void ReportRuns(const std::vector<Run>& runs) override {
        benchmark::ConsoleReporter::ReportRuns(runs);
        for (auto& r : runs) {
            results_.push_back(r);
            if (r.memory_result) {
                mem_results_.push_back(*r.memory_result);          // deep copy
                results_.back().memory_result = &mem_results_.back();
            }
        }
    }
    const std::vector<Run>& results() const { return results_; }
private:
    std::vector<Run>                             results_;
    std::deque<benchmark::MemoryManager::Result> mem_results_;  // deque: push_back never invalidates existing pointers
};

int main(int argc, char** argv) {
    // Scan for --plot[=DIR] and --report[=DIR] before passing argv to Google Benchmark.
    bool        do_plot      = false;
    std::string plot_dir     = "output/plots";
    bool        do_report    = false;
    std::string report_dir   = "output/plots";
    bool        do_memreport = false;

    std::vector<char*> gb_argv;
    gb_argv.push_back(argv[0]);
    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        if (arg == "--plot") {
            do_plot = true;
        } else if (arg.rfind("--plot=", 0) == 0) {
            do_plot  = true;
            plot_dir = arg.substr(7);
        } else if (arg == "--report") {
            do_report = true;
        } else if (arg.rfind("--report=", 0) == 0) {
            do_report  = true;
            report_dir = arg.substr(9);
        } else if (arg == "--memreport") {
            do_memreport = true;
        } else {
            gb_argv.push_back(argv[i]);
        }
    }
    // --report wins over --plot if both are specified
    if (do_report) do_plot = false;
    int gb_argc = static_cast<int>(gb_argv.size());

    mem::install();
    benchmark::Initialize(&gb_argc, gb_argv.data());
    if (benchmark::ReportUnrecognizedArguments(gb_argc, gb_argv.data()))
        return 1;

    if (do_report) {
        fs::create_directories(report_dir);

        CollectingReporter reporter;
        benchmark::RunSpecifiedBenchmarks(&reporter);
        benchmark::Shutdown();

        std::cout << "\ngenerating PNG plots in " << report_dir << "/ ...\n";
        try {
            bench_plot::plot_all_png(reporter.results(), report_dir);
            std::cout << "report plots written to " << report_dir << "/\n";
        } catch (const std::exception& e) {
            std::cerr << "plot error: " << e.what() << '\n';
            return 1;
        }
    } else if (do_plot) {
        fs::create_directories(plot_dir);

        CollectingReporter reporter;
        benchmark::RunSpecifiedBenchmarks(&reporter);
        benchmark::Shutdown();

        std::cout << "\ngenerating plots in " << plot_dir << "/ ...\n";
        try {
            bench_plot::plot_all(reporter.results(), plot_dir);
            std::cout << "done.\n";
        } catch (const std::exception& e) {
            std::cerr << "plot error: " << e.what() << '\n';
            return 1;
        }
    } else if (do_memreport) {
        CollectingReporter reporter;
        benchmark::RunSpecifiedBenchmarks(&reporter);
        benchmark::Shutdown();
        mem::print_report(reporter.results(), std::cout);
    } else {
        benchmark::RunSpecifiedBenchmarks();
        benchmark::Shutdown();
    }
    return 0;
}
