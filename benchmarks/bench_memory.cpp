/// @file benchmarks/bench_memory.cpp
/// @brief Definitions for bench_memory.hpp — operator new/delete replacements
///        and Google Benchmark MemoryManager integration.
///
/// One subtle detail: sized operator delete (the two-argument form) receives
/// the original allocation size, which lets us decrement g_live accurately.
/// The unsized form falls through to free() without touching g_live — this
/// only happens for allocations made before g_active was set, so it is safe.

#include "bench_memory.hpp"
#include <benchmark/benchmark.h>
#include <algorithm>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <new>
#include <string>

// ---------------------------------------------------------------------------
// Atomic counters (defined here, declared extern in the header)
// ---------------------------------------------------------------------------
namespace mem::detail {
    std::atomic<int64_t> g_allocs{0};
    std::atomic<int64_t> g_bytes {0};
    std::atomic<int64_t> g_live  {0};
    std::atomic<int64_t> g_peak  {0};
    std::atomic<bool>    g_active{false};

    void reset() noexcept {
        g_allocs.store(0, std::memory_order_relaxed);
        g_bytes .store(0, std::memory_order_relaxed);
        g_live  .store(0, std::memory_order_relaxed);
        g_peak  .store(0, std::memory_order_relaxed);
    }
} // namespace mem::detail

// ---------------------------------------------------------------------------
// Global operator new / delete replacements
// ---------------------------------------------------------------------------
// These are the standard replacement forms (C++17 §21.6.2).
// They run for every heap allocation in the process, so the g_active gate
// ensures we only count during the benchmark window.

void* operator new(std::size_t size) {
    void* p = std::malloc(size == 0 ? 1 : size);
    if (!p) throw std::bad_alloc{};

    if (mem::detail::g_active.load(std::memory_order_relaxed)) {
        using namespace mem::detail;
        auto sz = static_cast<int64_t>(size);
        g_allocs.fetch_add(1,  std::memory_order_relaxed);
        g_bytes .fetch_add(sz, std::memory_order_relaxed);

        int64_t live = g_live.fetch_add(sz, std::memory_order_relaxed) + sz;
        // update peak with a CAS loop (only a few iterations at most)
        int64_t peak = g_peak.load(std::memory_order_relaxed);
        while (live > peak &&
               !g_peak.compare_exchange_weak(peak, live,
                   std::memory_order_relaxed, std::memory_order_relaxed))
        {}
    }
    return p;
}

void* operator new[](std::size_t size) {
    return ::operator new(size);          // delegate — counted once
}

// Sized delete: receives the original size — accurate g_live accounting.
void operator delete(void* p, std::size_t size) noexcept {
    if (mem::detail::g_active.load(std::memory_order_relaxed))
        mem::detail::g_live.fetch_sub(static_cast<int64_t>(size),
                                      std::memory_order_relaxed);
    std::free(p);
}

void operator delete[](void* p, std::size_t size) noexcept {
    ::operator delete(p, size);
}

// Unsized delete: no size info, so g_live is not updated.
// Acceptable because unsized delete only fires for allocations predating
// g_active (runtime startup, static initializers) — they weren't counted.
void operator delete(void* p) noexcept  { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }

// ---------------------------------------------------------------------------
// MemoryManager — called by Google Benchmark around each benchmark function
// ---------------------------------------------------------------------------
namespace {

class HeapTracker : public benchmark::MemoryManager {
public:
    void Start() override {
        mem::detail::reset();
        mem::detail::g_active.store(true, std::memory_order_seq_cst);
    }

    void Stop(Result& result) override {
        mem::detail::g_active.store(false, std::memory_order_seq_cst);
        result.num_allocs            = mem::detail::g_allocs.load(std::memory_order_relaxed);
        result.total_allocated_bytes = mem::detail::g_bytes .load(std::memory_order_relaxed);
        result.max_bytes_used        = mem::detail::g_peak  .load(std::memory_order_relaxed);
        // net = total_allocated - total_freed; tracked via sized delete
        result.net_heap_growth       = mem::detail::g_live  .load(std::memory_order_relaxed);
    }
};

// Kept alive for the lifetime of the process after install().
std::unique_ptr<HeapTracker> g_tracker;

} // anonymous namespace

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------
void mem::install() {
    if (g_tracker) return;           // idempotent
    g_tracker = std::make_unique<HeapTracker>();
    benchmark::RegisterMemoryManager(g_tracker.get());
}

// ---------------------------------------------------------------------------
// Memory report formatter
// ---------------------------------------------------------------------------
namespace {

/// Format a byte count as a human-readable string (B / KB / MB).
std::string fmt_bytes(int64_t b) {
    if (b < 0) return "  —";
    char buf[32];
    if (b < 1024)
        std::snprintf(buf, sizeof(buf), "%lld B", static_cast<long long>(b));
    else if (b < 1024 * 1024)
        std::snprintf(buf, sizeof(buf), "%.1f KB", b / 1024.0);
    else
        std::snprintf(buf, sizeof(buf), "%.2f MB", b / (1024.0 * 1024.0));
    return buf;
}

} // anonymous namespace

void mem::print_report(
    const std::vector<benchmark::BenchmarkReporter::Run>& results,
    std::ostream& out)
{
    using Run = benchmark::BenchmarkReporter::Run;
    const int64_t tombstone = benchmark::MemoryManager::TombstoneValue;

    // Filter to runs that carry memory data and are not aggregate rows.
    std::vector<const Run*> mem_runs;
    for (auto& r : results) {
        if (r.memory_result && r.aggregate_name.empty())
            mem_runs.push_back(&r);
    }
    if (mem_runs.empty()) {
        out << "[memreport] No memory data — was mem::install() called before "
               "benchmark::Initialize()?\n";
        return;
    }

    // Column widths
    std::size_t name_w = 9; // "Benchmark"
    for (auto* r : mem_runs)
        name_w = std::max(name_w, r->run_name.str().size());

    const int col = 14;
    const std::string sep(name_w + 4 * (col + 2) + 2, '-');

    out << "\n=== Memory Report ===\n\n";
    out << std::left  << std::setw(static_cast<int>(name_w)) << "Benchmark"
        << "  "
        << std::right << std::setw(col) << "allocs/iter"
        << "  "       << std::setw(col) << "bytes/iter"
        << "  "       << std::setw(col) << "peak live"
        << "  "       << std::setw(col) << "net growth"
        << "\n" << sep << "\n";

    for (auto* r : mem_runs) {
        auto& mr = *r->memory_result;
        double allocs = r->allocs_per_iter;

        // bytes/iter: total_allocated_bytes / memory_iterations.
        // memory_iterations = min(16, benchmark_iters); we recover it via
        // allocs_per_iter = num_allocs / memory_iterations.
        // Rather than recompute, we show total_allocated_bytes / num_allocs * allocs_per_iter
        // which simplifies to: total_allocated_bytes / memory_iterations.
        // Just show the raw totals alongside per-iter allocs for clarity.
        int64_t total = (mr.total_allocated_bytes == tombstone) ? -1 : mr.total_allocated_bytes;
        int64_t peak  = (mr.max_bytes_used        == tombstone) ? -1 : mr.max_bytes_used;
        int64_t net   = (mr.net_heap_growth       == tombstone) ? -1 : mr.net_heap_growth;

        // Compute per-iter bytes (divide total by memory_iterations).
        // memory_iterations = num_allocs / allocs_per_iter  (if allocs_per_iter > 0).
        std::string bytes_col;
        if (allocs > 0.0 && total >= 0 && mr.num_allocs > 0) {
            double mem_iters = static_cast<double>(mr.num_allocs) / allocs;
            double bytes_per_iter = static_cast<double>(total) / mem_iters;
            char buf[32];
            if (bytes_per_iter < 1024)
                std::snprintf(buf, sizeof(buf), "%.0f B", bytes_per_iter);
            else if (bytes_per_iter < 1024 * 1024)
                std::snprintf(buf, sizeof(buf), "%.1f KB", bytes_per_iter / 1024.0);
            else
                std::snprintf(buf, sizeof(buf), "%.2f MB", bytes_per_iter / (1024.0 * 1024.0));
            bytes_col = buf;
        } else {
            bytes_col = (total == 0) ? "0 B" : fmt_bytes(total);
        }

        // Annotation: highlight allocation-free hot paths
        std::string note = (allocs < 0.01) ? "  [alloc-free]" : "";

        out << std::left  << std::setw(static_cast<int>(name_w)) << r->run_name.str()
            << "  "
            << std::right << std::setw(col) << std::fixed << std::setprecision(2) << allocs
            << "  "       << std::setw(col) << bytes_col
            << "  "       << std::setw(col) << fmt_bytes(peak)
            << "  "       << std::setw(col) << fmt_bytes(net)
            << note << "\n";
    }
    out << sep << "\n\n";
    out << "Notes:\n"
        << "  allocs/iter  : heap allocations per benchmark iteration (setup + hot path)\n"
        << "  bytes/iter   : total bytes requested from allocator per iteration\n"
        << "  peak live    : maximum simultaneously live heap during the run\n"
        << "  net growth   : bytes still allocated after the run (tracked via sized delete)\n"
        << "  [alloc-free] : hot path never touched the heap — ideal for production kernels\n\n";
}
