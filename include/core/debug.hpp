/// @file core/debug.hpp
/// @brief Diagnostic presets, failure reporting, and dimension checks.
///
/// The scaffolding every module reports through. Sampling of mathematical
/// properties builds on this and lives in `algebra/debug.hpp`.
#pragma once

#include "core/types.hpp"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <concepts>
#include <cstdint>
#include <iostream>
#include <limits>
#include <source_location>
#include <stdexcept>
#include <string>
#include <string_view>

namespace num::debug {

enum class diagnostic_level {
    off = 0,
    basic = 1, ///< Dimension matching, non-empty, finite checks
    full = 2   ///< Includes sampled symmetry, adjoint, and positive-definiteness testing
};

/// @brief High-level execution and diagnostic presets for mathematical invariant enforcement.
enum class diagnostic_preset : std::uint8_t {
    strict,   ///< Full runtime property sampling (x^T A x > 0, symmetry), loud warnings on untagged
              ///< inputs.
    balanced, ///< Dimension matching, structure bounds, warnings on untagged inputs (skips
              ///< randomized sampling).
    unsafe,   ///< Prototyping mode: completely silences unverified invariant warnings and property
              ///< checks.
    production, ///< Maximum throughput: all diagnostics disabled.
};

namespace preset {
inline constexpr diagnostic_preset strict = diagnostic_preset::strict;
inline constexpr diagnostic_preset balanced = diagnostic_preset::balanced;
inline constexpr diagnostic_preset unsafe = diagnostic_preset::unsafe;
inline constexpr diagnostic_preset prototype = diagnostic_preset::unsafe;
inline constexpr diagnostic_preset production = diagnostic_preset::production;
} // namespace preset

} // namespace num::debug

// -----------------------------------------------------------------------------
// The compile-time ceiling
// -----------------------------------------------------------------------------
//
// `NUMERICS_DIAGNOSTICS` bounds what this build can do; the runtime preset selects from
// what is left. The two answer different questions. The ceiling decides what code exists,
// the preset decides whether it runs.
//
//   0  nothing. Every check and probe is discarded at compile time.
//   1  shape checks. Dimensions, emptiness, finiteness. O(1) or O(n), and worth keeping.
//   2  property sampling as well. Symmetry, definiteness and linearity probes over random
//      vectors, which cost O(n^2) in the operator's size.
//
// The default is 2 without NDEBUG and 1 with it. Level 2 is not a Release default because
// it is not a small cost: attaching an SPD claim to a 1024x1024 operator sampled at about
// 15 ms against a 1.5 ms conjugate-gradient solve of the same system. A default that
// makes the assertion ten times more expensive than the work it guards is one that gets
// measured and blamed on the library.
//
// Override it explicitly. `-DNUMERICS_DIAGNOSTICS=2` keeps full sampling in an optimized
// build, which is what a test suite or a numerically suspicious run wants.
#if !defined(NUMERICS_DIAGNOSTICS)
#if defined(NDEBUG)
#define NUMERICS_DIAGNOSTICS 1
#else
#define NUMERICS_DIAGNOSTICS 2
#endif
#endif

#if NUMERICS_DIAGNOSTICS < 0 || NUMERICS_DIAGNOSTICS > 2
#error "NUMERICS_DIAGNOSTICS must be 0 (off), 1 (shape checks), or 2 (property sampling)"
#endif

namespace num::debug {

/// @brief The strongest level this build can reach. The runtime preset is clamped to it.
inline constexpr diagnostic_level compiled_level =
    static_cast<diagnostic_level>(NUMERICS_DIAGNOSTICS);

/// @brief Whether shape, emptiness and finiteness checks exist in this build.
inline constexpr bool checks_compiled_in = NUMERICS_DIAGNOSTICS >= 1;

/// @brief Whether property sampling exists in this build.
///
/// Guard a probe with `if constexpr (!sampling_compiled_in) { return; }` and its body is
/// discarded rather than branched over. This is what makes `preset::production` remove
/// the code, which the documentation claimed long before it was true.
inline constexpr bool sampling_compiled_in = NUMERICS_DIAGNOSTICS >= 2;

namespace detail {

constexpr diagnostic_level level_for(diagnostic_preset p) noexcept {
    switch (p) {
    case diagnostic_preset::strict:
        return diagnostic_level::full;
    case diagnostic_preset::balanced:
        return diagnostic_level::basic;
    case diagnostic_preset::unsafe:
    case diagnostic_preset::production:
        break;
    }
    return diagnostic_level::off;
}

constexpr diagnostic_level clamp_to_ceiling(diagnostic_level lvl) noexcept {
    return static_cast<int>(lvl) > NUMERICS_DIAGNOSTICS ? compiled_level : lvl;
}

constexpr diagnostic_preset default_preset() noexcept {
    return compiled_level == diagnostic_level::full    ? diagnostic_preset::strict
           : compiled_level == diagnostic_level::basic ? diagnostic_preset::balanced
                                                       : diagnostic_preset::production;
}

// Read on every check and written rarely, so a relaxed atomic is a plain load on any real
// target. It is atomic rather than a bare global because `scoped_preset` makes it
// reachable from more than one thread, and a torn read there would be undefined behaviour
// for the sake of nothing.
inline std::atomic<diagnostic_level> g_level{clamp_to_ceiling(level_for(default_preset()))};
inline std::atomic<diagnostic_preset> g_preset{default_preset()};

} // namespace detail

inline void set_level(diagnostic_level lvl) noexcept {
    detail::g_level.store(detail::clamp_to_ceiling(lvl), std::memory_order_relaxed);
}

[[nodiscard]] inline diagnostic_level get_level() noexcept {
    return detail::g_level.load(std::memory_order_relaxed);
}

/// @brief Configure the global execution preset.
///
/// The request is clamped to the compile-time ceiling. Asking for `strict` in a build
/// configured at level 1 leaves sampling off, because that code was never emitted.
/// `get_preset` still reports what was asked for; `preset_fully_applied` reports whether
/// the request was met.
inline void set_preset(diagnostic_preset p) noexcept {
    detail::g_preset.store(p, std::memory_order_relaxed);
    detail::g_level.store(detail::clamp_to_ceiling(detail::level_for(p)),
                          std::memory_order_relaxed);
}

/// @brief The preset most recently requested.
///
/// Stored rather than derived from the level. Deriving it could not tell `unsafe` from
/// `production`, which share a level, so `set_preset(production)` used to report back
/// `unsafe`.
[[nodiscard]] inline diagnostic_preset get_preset() noexcept {
    return detail::g_preset.load(std::memory_order_relaxed);
}

/// @brief Whether the preset in force is also reachable in this build.
///
/// False when a caller asked for more than the ceiling allows. Worth reporting once at
/// startup rather than silently running fewer checks than the caller believes.
[[nodiscard]] inline bool preset_fully_applied() noexcept {
    return detail::level_for(get_preset()) == get_level();
}

/// @brief RAII guard to temporarily apply an execution preset within a scope.
class scoped_preset {
  public:
    explicit scoped_preset(diagnostic_preset temp_preset) noexcept
        : previous_preset_(get_preset()) {
        set_preset(temp_preset);
    }

    ~scoped_preset() noexcept { set_preset(previous_preset_); }

    scoped_preset(const scoped_preset &) = delete;
    scoped_preset &operator=(const scoped_preset &) = delete;
    scoped_preset(scoped_preset &&) = delete;
    scoped_preset &operator=(scoped_preset &&) = delete;

  private:
    diagnostic_preset previous_preset_;
};

/// @brief Raise a descriptive diagnostic exception with source location info.
[[noreturn]] inline void panic(std::string_view category, std::string_view message,
                               std::source_location loc = std::source_location::current()) {
    std::string err = "[" + std::string(category) + "] Error at " + loc.file_name() + ":" +
                      std::to_string(loc.line()) + " in " + loc.function_name() + ":\n  " +
                      std::string(message);
    throw std::invalid_argument(err);
}

/// @brief Verify dimension equality (e.g. A.rows() == b.size())
inline void check_dim(idx expected, idx actual, std::string_view label,
                      std::source_location loc = std::source_location::current()) {
    if constexpr (checks_compiled_in) {
        if (get_level() != diagnostic_level::off && expected != actual) {
            panic("DimensionError",
                  "Dimension mismatch for " + std::string(label) + ": expected " +
                      std::to_string(expected) + ", got " + std::to_string(actual),
                  loc);
        }
    }
}

/// @brief Verify a container is non-empty
inline void check_non_empty(idx size, std::string_view label,
                            std::source_location loc = std::source_location::current()) {
    if constexpr (checks_compiled_in) {
        if (get_level() != diagnostic_level::off && size == 0) {
            panic("ValueError", std::string(label) + " cannot be empty (size is 0)", loc);
        }
    }
}

/// @brief Verify all values in array are finite (not NaN or Inf)
template <typename T>
inline void check_finite(const T *data, idx n, std::string_view label,
                         std::source_location loc = std::source_location::current()) {
    if constexpr (checks_compiled_in) {
        if (get_level() == diagnostic_level::off) {
            return;
        }
        for (idx i = 0; i < n; ++i) {
            if (!std::isfinite(data[i])) {
                panic("ValueError",
                      std::string(label) + " contains non-finite value (NaN or Inf) at index " +
                          std::to_string(i),
                      loc);
            }
        }
    }
}
} // namespace num::debug

namespace num {

using debug::diagnostic_preset;
using debug::get_preset;
using debug::preset_fully_applied;
using debug::scoped_preset;
using debug::set_preset;
namespace preset = debug::preset;

} // namespace num
