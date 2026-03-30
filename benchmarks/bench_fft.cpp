/// @file bench_fft.cpp
/// @brief Backend-comparative benchmarks for num::spectral FFT.
///
/// Benchmarks (templated on FFTBackend):
///   BM_FFT      -- one-shot forward complex DFT
///   BM_IFFT     -- one-shot inverse complex DFT
///   BM_RFFT     -- real-to-complex forward DFT
///   BM_FFTPlan  -- reusable plan (plan creation excluded from timing)
///
/// Backends under test:
///   seq     -- scalar Cooley-Tukey radix-2 DIT, precomputed twiddles
///   simd    -- handwritten AVX2 / NEON butterfly (2 pairs/cycle)
///   stdsimd -- std::experimental::simd butterfly (GCC 11+ only)
///   fftw    -- FFTW3 mixed-radix with SIMD codelets (when available)
///
/// Counter: bytes_per_second reported as GB/s
///   (reads 16n + writes 16n bytes for complex DFT)

#include <benchmark/benchmark.h>
#include "spectral/fft.hpp"
#include "core/types.hpp"
#include <cmath>

using namespace num;
using namespace num::spectral;

static constexpr double TWO_PI = 6.283185307179586476925286766559;

static void fill_signal(CVector& v) {
    const int n = static_cast<int>(v.size());
    for (int j = 0; j < n; ++j)
        v[j] = std::complex<double>{
            std::sin(TWO_PI * 7  * j / n) + std::cos(TWO_PI * 13 * j / n),
            std::cos(TWO_PI * 11 * j / n) - std::sin(TWO_PI * 19 * j / n)
        };
}

static void fill_real_signal(Vector& v) {
    const int n = static_cast<int>(v.size());
    for (int j = 0; j < n; ++j)
        v[j] = std::sin(TWO_PI * 7 * j / n) + 0.5 * std::cos(TWO_PI * 23 * j / n);
}

// -- BM_FFT: one-shot forward complex DFT ------------------------------------

template<FFTBackend B>
static void BM_FFT(benchmark::State& state) {
    const idx n = static_cast<idx>(state.range(0));
    CVector in(n), out(n);
    fill_signal(in);
    for (auto _ : state) {
        fft(in, out, B);
        benchmark::DoNotOptimize(out.data());
    }
    state.SetBytesProcessed(
        state.iterations() * static_cast<int64_t>(n) * 2 * sizeof(std::complex<double>));
    state.SetComplexityN(static_cast<int64_t>(n));
}

BENCHMARK_TEMPLATE(BM_FFT, FFTBackend::seq)
    ->RangeMultiplier(4)->Range(256, 1 << 18)->Complexity();
BENCHMARK_TEMPLATE(BM_FFT, FFTBackend::simd)
    ->RangeMultiplier(4)->Range(256, 1 << 18)->Complexity();
#ifdef NUMERICS_HAS_STD_SIMD
BENCHMARK_TEMPLATE(BM_FFT, FFTBackend::stdsimd)
    ->RangeMultiplier(4)->Range(256, 1 << 18)->Complexity();
#endif
#ifdef NUMERICS_HAS_FFTW
BENCHMARK_TEMPLATE(BM_FFT, FFTBackend::fftw)
    ->RangeMultiplier(4)->Range(256, 1 << 20)->Complexity();
#endif

// -- BM_IFFT: one-shot inverse complex DFT -----------------------------------

template<FFTBackend B>
static void BM_IFFT(benchmark::State& state) {
    const idx n = static_cast<idx>(state.range(0));
    CVector in(n), out(n);
    fill_signal(in);
    for (auto _ : state) {
        ifft(in, out, B);
        benchmark::DoNotOptimize(out.data());
    }
    state.SetBytesProcessed(
        state.iterations() * static_cast<int64_t>(n) * 2 * sizeof(std::complex<double>));
    state.SetComplexityN(static_cast<int64_t>(n));
}

BENCHMARK_TEMPLATE(BM_IFFT, FFTBackend::seq)
    ->RangeMultiplier(4)->Range(256, 1 << 18)->Complexity();
BENCHMARK_TEMPLATE(BM_IFFT, FFTBackend::simd)
    ->RangeMultiplier(4)->Range(256, 1 << 18)->Complexity();
#ifdef NUMERICS_HAS_STD_SIMD
BENCHMARK_TEMPLATE(BM_IFFT, FFTBackend::stdsimd)
    ->RangeMultiplier(4)->Range(256, 1 << 18)->Complexity();
#endif
#ifdef NUMERICS_HAS_FFTW
BENCHMARK_TEMPLATE(BM_IFFT, FFTBackend::fftw)
    ->RangeMultiplier(4)->Range(256, 1 << 20)->Complexity();
#endif

// -- BM_RFFT: real-to-complex forward DFT ------------------------------------

template<FFTBackend B>
static void BM_RFFT(benchmark::State& state) {
    const idx n = static_cast<idx>(state.range(0));
    Vector in(n);
    CVector out(static_cast<idx>(n / 2 + 1));
    fill_real_signal(in);
    for (auto _ : state) {
        rfft(in, out, B);
        benchmark::DoNotOptimize(out.data());
    }
    state.SetBytesProcessed(
        state.iterations() * (
            static_cast<int64_t>(n) * sizeof(double) +
            static_cast<int64_t>(n / 2 + 1) * sizeof(std::complex<double>)
        ));
    state.SetComplexityN(static_cast<int64_t>(n));
}

BENCHMARK_TEMPLATE(BM_RFFT, FFTBackend::seq)
    ->RangeMultiplier(4)->Range(256, 1 << 18)->Complexity();
BENCHMARK_TEMPLATE(BM_RFFT, FFTBackend::simd)
    ->RangeMultiplier(4)->Range(256, 1 << 18)->Complexity();
#ifdef NUMERICS_HAS_STD_SIMD
BENCHMARK_TEMPLATE(BM_RFFT, FFTBackend::stdsimd)
    ->RangeMultiplier(4)->Range(256, 1 << 18)->Complexity();
#endif
#ifdef NUMERICS_HAS_FFTW
BENCHMARK_TEMPLATE(BM_RFFT, FFTBackend::fftw)
    ->RangeMultiplier(4)->Range(256, 1 << 20)->Complexity();
#endif

// -- BM_FFTPlan: reusable plan (plan creation excluded) ----------------------

template<FFTBackend B>
static void BM_FFTPlan(benchmark::State& state) {
    const idx n = static_cast<idx>(state.range(0));
    CVector in(n), out(n);
    fill_signal(in);
    FFTPlan plan(static_cast<int>(n), true, B);

    for (auto _ : state) {
        plan.execute(in, out);
        benchmark::DoNotOptimize(out.data());
    }
    state.SetBytesProcessed(
        state.iterations() * static_cast<int64_t>(n) * 2 * sizeof(std::complex<double>));
    state.SetComplexityN(static_cast<int64_t>(n));
}

BENCHMARK_TEMPLATE(BM_FFTPlan, FFTBackend::seq)
    ->RangeMultiplier(4)->Range(256, 1 << 18)->Complexity();
BENCHMARK_TEMPLATE(BM_FFTPlan, FFTBackend::simd)
    ->RangeMultiplier(4)->Range(256, 1 << 18)->Complexity();
#ifdef NUMERICS_HAS_STD_SIMD
BENCHMARK_TEMPLATE(BM_FFTPlan, FFTBackend::stdsimd)
    ->RangeMultiplier(4)->Range(256, 1 << 18)->Complexity();
#endif
#ifdef NUMERICS_HAS_FFTW
BENCHMARK_TEMPLATE(BM_FFTPlan, FFTBackend::fftw)
    ->RangeMultiplier(4)->Range(256, 1 << 20)->Complexity();
#endif
