/// @file benchmarks/bench_memory.hpp
/// @brief Heap-allocation tracker for Google Benchmark.
///
/// Replaces global operator new/delete with atomic counters and registers
/// a benchmark::MemoryManager so every benchmark automatically reports:
///
///   allocs/iter   — heap allocations per benchmark iteration
///   bytes/iter    — bytes allocated per iteration  (maps to net_heap_growth)
///   peak_bytes    — maximum live heap bytes at any point  (maps to max_bytes_used)
///
/// Usage (main.cpp):
///   #include "bench_memory.hpp"
///   int main(...) {
///       mem::install();          // register before RunSpecifiedBenchmarks
///       benchmark::Initialize(&argc, argv);
///       benchmark::RunSpecifiedBenchmarks();
///       benchmark::Shutdown();
///   }
///
/// The operator new/delete replacements live in bench_memory.cpp (single TU).
/// This header only declares the public interface.

#pragma once
#include <benchmark/benchmark.h>
#include <atomic>
#include <cstddef>
#include <ostream>
#include <vector>

namespace mem {

/// Atomic counters written by the operator new/delete replacements.
/// Exposed here so bench_memory.cpp can define them exactly once while
/// other TUs that include this header can read them (via mem::install).
namespace detail {
    extern std::atomic<int64_t> g_allocs;   ///< allocation count since last reset
    extern std::atomic<int64_t> g_bytes;    ///< total bytes allocated since last reset
    extern std::atomic<int64_t> g_live;     ///< current live bytes (alloc - freed)
    extern std::atomic<int64_t> g_peak;     ///< max live bytes seen since last reset
    extern std::atomic<bool>    g_active;   ///< gate: counters only increment when true

    void reset() noexcept;
} // namespace detail

/// Register the MemoryManager with Google Benchmark.
/// Must be called before benchmark::Initialize / RunSpecifiedBenchmarks.
/// Safe to call multiple times (idempotent after first call).
void install();

/// Print a human-readable memory comparison table to `out`.
///
/// Columns shown (only for runs that have memory_result):
///   allocs/iter   — heap allocations per benchmark iteration
///   bytes/iter    — total bytes allocated per iteration
///   peak bytes    — maximum live heap at any instant during the run
///   net bytes     — bytes still live after the run (leaked or intentional)
///
/// Runs with allocs/iter == 0 are flagged "allocation-free" — the hot path
/// never touched the heap, which is the ideal result for production kernels.
///
/// @param results  Vector of Run objects from CollectingReporter::results()
/// @param out      Output stream (default std::cout)
void print_report(
    const std::vector<benchmark::BenchmarkReporter::Run>& results,
    std::ostream& out);

} // namespace mem
