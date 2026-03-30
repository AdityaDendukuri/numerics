# Benchmarks

This directory contains Google Benchmark microbenchmarks for every module in the library.
It is also the place to add memory analysis work: the runner ships built-in heap tracking
via a custom `MemoryManager`, and the build supports sanitizer and Valgrind workflows.

---

## Quick start

```bash
# Standard build (Release)
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target numerics_bench -j$(nproc)

# Run everything
./build/benchmarks/numerics_bench

# Filter to one module
./build/benchmarks/numerics_bench --benchmark_filter=BM_Matmul

# Memory comparison table (see below)
./build/benchmarks/numerics_bench --benchmark_filter=BM_Matmul --memreport
```

---

## Memory analysis tools

Three complementary tools are available, each answering a different question:

| Tool | Question answered | When to use |
|------|-------------------|-------------|
| `--memreport` | *How much* does each backend allocate? | Comparing backends, profiling hot paths |
| `-DNUMERICS_SANITIZE=asan` | Is there a heap **bug**? (overflow, use-after-free, leak) | Development, CI |
| `valgrind --tool=memcheck` | Same as ASan but no recompile; more edge cases | Debugging third-party or pre-built code |

The three are complementary: `--memreport` measures quantity; ASan/Valgrind detect correctness bugs.

---

## `--memreport`: built-in heap allocation profiling

The runner's `MemoryManager` replaces the global `operator new`/`delete` with atomic
counters, then asks Google Benchmark to run each benchmark for a fixed 16-iteration
memory pass before reporting.  Every benchmark automatically gets four extra columns:

| Column | What it shows |
|--------|---------------|
| `allocs/iter` | Heap allocations per benchmark iteration (setup + hot path) |
| `bytes/iter` | Total bytes requested from the allocator per iteration |
| `peak live` | Maximum simultaneously live heap bytes during the run |
| `net growth` | Bytes still allocated after the run (tracked via sized `delete`) |

A run annotated `[alloc-free]` has `allocs/iter < 0.01`; the hot path never touched the
heap.  This is the ideal result for any production kernel.

### Example: matrix multiplication backends (N = 128)

```
./build/benchmarks/numerics_bench \
    --benchmark_filter="BM_Matmul.*/128" --memreport
```

```
Benchmark                           allocs/iter      bytes/iter       peak live      net growth
-------------------------------------------------------------------------------------------------
BM_Matmul_Naive/128                        0.44         24.0 KB        384.6 KB        384.6 KB
BM_Matmul_Blocked/128                      0.50         24.0 KB        384.6 KB        384.6 KB
BM_Matmul_RegBlocked/128                   0.50         24.0 KB        384.6 KB        384.6 KB
BM_Matmul<Backend::blocked>/128            0.50         24.0 KB        384.6 KB        384.6 KB
BM_Matmul<Backend::simd>/128               0.50         24.0 KB        384.6 KB        384.6 KB
BM_Matmul<Backend::blas>/128               0.50         24.0 KB        384.6 KB        384.6 KB
BM_Matmul<Backend::omp>/128                0.50         24.0 KB        384.6 KB        384.6 KB
BM_Matmul_Scalar/128                       0.50         24.0 KB        384.6 KB        384.6 KB
BM_Matmul_Scalar_Blocked/128               0.50         24.0 KB        384.6 KB        384.6 KB
-------------------------------------------------------------------------------------------------
```

**Reading the table:**

- `allocs/iter ≈ 0.5` means roughly half an allocation per iteration.
  Google Benchmark's memory pass runs 16 iterations of the benchmark function.
  `0.5 × 16 = 8` total allocations — these are the three `Matrix` objects
  (`A`, `B`, `C`) constructed *once before the timing loop*, plus a few bytes of
  benchmark-internal overhead.
- `bytes/iter = 24.0 KB` = 393 KB ÷ 16 iterations.  Three 128×128 `double`
  matrices × 131 072 bytes each = 393 216 bytes, distributed across 16 runs.
- **The hot path is allocation-free.**  No `new`/`delete` inside the timing loop —
  all backends are identical here because the matrices were pre-allocated.

This is the *correct* pattern for an HPC kernel: setup allocates once, the loop
computes with existing buffers.

### Comparison: solvers that allocate proportional to problem size

```
./build/benchmarks/numerics_bench \
    --benchmark_filter="BM_CG/|BM_Thomas/" --memreport
```

```
Benchmark           allocs/iter      bytes/iter       peak live      net growth
BM_CG/32                   3.44          1.3 KB         21.1 KB         21.1 KB
BM_CG/64                   3.44          3.6 KB         57.6 KB         57.6 KB
BM_CG/128                  3.44         11.2 KB        178.6 KB        178.6 KB
BM_CG/256                  3.44         38.3 KB        612.6 KB        612.6 KB
BM_Thomas/64               2.69          1.2 KB         19.1 KB         19.1 KB
BM_Thomas/256              2.69          4.7 KB         74.6 KB         74.6 KB
BM_Thomas/1024             2.69         18.5 KB        296.6 KB        296.6 KB
BM_Thomas/4096             2.69         74.0 KB         1.16 MB         1.16 MB
```

`bytes/iter` scales linearly with `N`.  The allocations again come from the setup
vectors (`a`, `b`, `c`, `d`, `x` for Thomas; four work vectors inside CG), not
from the algorithms themselves.  Both are allocation-free in their hot paths.

### What to watch for when parallelizing

When you implement a parallel backend (OpenMP / CUDA stub), run `--memreport`
before and after.  Common mistakes that show up:

| Symptom | Likely cause |
|---------|-------------|
| `allocs/iter` jumps by `num_threads` | Temporary buffer allocated *inside* the parallel region per thread — move it outside |
| `bytes/iter` grows with thread count | Same issue: per-thread scratch space leaking into the hot path |
| `allocs/iter ≫ 1` for a formerly allocation-free kernel | `std::vector` copy inside a lambda capture; use `std::span` or a reference instead |
| `peak live` >> `net growth` | Memory allocated and freed inside the loop — no leak, but pressure can hurt the allocator under concurrency |

---

## Sanitizer builds: catching heap bugs

The CMake option `-DNUMERICS_SANITIZE` injects compiler-level instrumentation that
catches memory bugs at runtime.  Use a *separate* build directory to avoid
polluting the production build.

```bash
# AddressSanitizer: heap buffer overflow, use-after-free, heap leak
cmake -B build-asan \
      -DCMAKE_BUILD_TYPE=RelWithDebInfo \
      -DNUMERICS_SANITIZE=asan \
      -DNUMERICS_ENABLE_CUDA=OFF      # ASan and CUDA runtime conflict
cmake --build build-asan -j$(nproc)

# Run the test suite under ASan
ctest --test-dir build-asan

# Run a specific benchmark (slower — ASan adds ~2× overhead)
./build-asan/benchmarks/numerics_bench --benchmark_filter=BM_CG

# UndefinedBehaviorSanitizer: signed overflow, null deref, misaligned access
cmake -B build-ubsan \
      -DCMAKE_BUILD_TYPE=RelWithDebInfo \
      -DNUMERICS_SANITIZE=ubsan
cmake --build build-ubsan -j$(nproc)
ctest --test-dir build-ubsan

# ThreadSanitizer: data races — use when testing your OpenMP/MPI parallel code
cmake -B build-tsan \
      -DCMAKE_BUILD_TYPE=RelWithDebInfo \
      -DNUMERICS_SANITIZE=tsan \
      -DNUMERICS_USE_OPENMP=ON
cmake --build build-tsan -j$(nproc)
ctest --test-dir build-tsan

# Combine sanitizers (asan + ubsan catch the most in one pass)
cmake -B build-san \
      -DCMAKE_BUILD_TYPE=RelWithDebInfo \
      -DNUMERICS_SANITIZE=asan,ubsan
```

**When ASan fires you will see:**

```
==12345==ERROR: AddressSanitizer: heap-buffer-overflow on address 0x...
READ of size 8 at 0x... thread T0
    #0 0x... in num::matmul(Matrix const&, ...) bench_linalg.cpp:52
    #1 0x... in BM_Matmul_Naive ...
```

The stack trace points directly to the offending line.  Fix the bug, re-run
`ctest --test-dir build-asan` to confirm all tests are clean.

### ASan vs Valgrind

| | AddressSanitizer | Valgrind memcheck |
|-|-----------------|-------------------|
| Overhead | ~2× slowdown | ~20× slowdown |
| Requires recompile | Yes (`-fsanitize=address`) | No |
| Stack trace quality | Excellent (inline) | Good |
| Detects uninitialised reads | No (use MSan) | Yes |
| Works with CUDA | No | Partial |
| Best for | Development workflow | Pre-built or third-party code |

---

## Valgrind

Valgrind does not require recompilation.  Build in Debug or RelWithDebInfo, then
prefix the binary.

```bash
cmake -B build-dbg -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build-dbg --target numerics_bench -j$(nproc)

# Leak check with full origins
valgrind \
    --tool=memcheck \
    --leak-check=full \
    --track-origins=yes \
    --error-exitcode=1 \
    ./build-dbg/benchmarks/numerics_bench \
        --benchmark_filter=BM_CG/32 \
        --benchmark_min_time=0.01s

# Heap profiler (call-graph allocation sizes over time)
valgrind \
    --tool=massif \
    --pages-as-heap=no \
    ./build-dbg/benchmarks/numerics_bench \
        --benchmark_filter=BM_Matmul_Naive/128 \
        --benchmark_min_time=0.1s
ms_print massif.out.* | head -60

# Cache and branch predictor simulation
valgrind \
    --tool=cachegrind \
    --cache-sim=yes \
    --branch-sim=yes \
    ./build-dbg/benchmarks/numerics_bench \
        --benchmark_filter=BM_Matmul_Blocked/128
cg_annotate cachegrind.out.*
```

**Tip:** pass `--benchmark_min_time=0.01s` to reduce the number of iterations;
Valgrind's 20× overhead makes full benchmark runs impractical.  Just enough
iterations to trigger the code paths you care about is sufficient for leak checking.

---

## Machine-readable output

For scripting or notebook post-processing, use JSON output:

```bash
./build/benchmarks/numerics_bench \
    --benchmark_filter="BM_Matmul.*/128" \
    --benchmark_format=json \
    > results.json
```

The JSON contains all four memory fields per run when a `MemoryManager` is
registered:

```json
{
  "name": "BM_Matmul_Naive/128",
  "allocs_per_iter": 0.4375,
  "total_allocated_bytes": 393784,
  "max_bytes_used": 393784,
  "net_heap_growth": 393784
}
```

Parse with Python:

```python
import json

with open("results.json") as f:
    data = json.load(f)

for b in data["benchmarks"]:
    if "allocs_per_iter" in b:
        print(f"{b['name']:<45} "
              f"allocs={b['allocs_per_iter']:.2f}  "
              f"bytes={b.get('total_allocated_bytes', 'N/A')}")
```

---

## Adding your own benchmark

Copy the pattern from `bench_linalg.cpp`:

```cpp
static void BM_MyKernel(benchmark::State& state) {
    idx n = static_cast<idx>(state.range(0));

    // Setup: allocate outside the loop — this is NOT in the hot path.
    MyType data(n);

    for (auto _ : state) {
        my_kernel(data);
        benchmark::DoNotOptimize(data.result());
    }

    // Optional: report throughput or FLOP/s
    state.SetBytesProcessed(
        state.iterations() * n * sizeof(double));
    state.SetComplexityN(static_cast<int64_t>(n));
}
BENCHMARK(BM_MyKernel)->RangeMultiplier(2)->Range(64, 4096)->Complexity();
```

Then add the `.cpp` file to `CMakeLists.txt`:

```cmake
add_executable(numerics_bench
    main.cpp
    bench_memory.cpp
    bench_linalg.cpp
    bench_banded.cpp
    bench_autovec.cpp
    bench_fft.cpp
    bench_mymodule.cpp   # ← new file
)
```

Run with `--memreport` to confirm your kernel is allocation-free in the hot path
before submitting a parallel implementation.
