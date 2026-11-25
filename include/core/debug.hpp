/// @file core/debug.hpp
/// @brief Diagnostic presets, failure reporting and dimension checks.
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

/// @brief Diagnostic preset: how much runtime checking runs.
enum class diagnostic_preset : std::uint8_t {
    strict,     ///< Shape checks and property sampling.
    balanced,   ///< Shape checks only.
    unsafe,     ///< Checks silenced.
    production, ///< Diagnostics compiled out.
};

namespace preset {
/// @brief Preset: shape checks and property sampling.
inline constexpr diagnostic_preset strict = diagnostic_preset::strict;
/// @brief Preset: shape checks only.
inline constexpr diagnostic_preset balanced = diagnostic_preset::balanced;
/// @brief Preset: checks silenced.
inline constexpr diagnostic_preset unsafe = diagnostic_preset::unsafe;
/// @brief Preset: alias of `unsafe`.
inline constexpr diagnostic_preset prototype = diagnostic_preset::unsafe;
/// @brief Preset: diagnostics compiled out.
inline constexpr diagnostic_preset production = diagnostic_preset::production;
} // namespace preset

} // namespace num::debug

// The compile-time ceiling
//
// `NUMERICS_DIAGNOSTICS` decides what checks exist; the runtime preset decides which run.
//
//   0  nothing.
//   1  shape checks: dimensions, emptiness, finiteness.
//   2  property sampling as well, O(n^2) in the operator size.
//
// The default is 2 without NDEBUG and 1 with it. Sampling an SPD claim on a 1024x1024
// operator took about 15 ms against a 1.5 ms CG solve, so it is off in Release unless set.
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

/// @brief Highest diagnostic level compiled into this build.
inline constexpr diagnostic_level compiled_level =
    static_cast<diagnostic_level>(NUMERICS_DIAGNOSTICS);

/// @brief Whether shape, emptiness and finiteness checks are compiled in.
inline constexpr bool checks_compiled_in = NUMERICS_DIAGNOSTICS >= 1;

/// @brief Whether property sampling is compiled in.
/// @note Guard probes with `if constexpr (!sampling_compiled_in)` so production drops them.
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

/// @brief Set the diagnostic level, clamped to `compiled_level`.
inline void set_level(diagnostic_level lvl) noexcept {
    detail::g_level.store(detail::clamp_to_ceiling(lvl), std::memory_order_relaxed);
}

/// @brief Current diagnostic level.
[[nodiscard]] inline diagnostic_level get_level() noexcept {
    return detail::g_level.load(std::memory_order_relaxed);
}

/// @brief Set the diagnostic preset, clamped to `compiled_level`.
/// @note A preset above the compiled level runs fewer checks than requested; see
/// `preset_fully_applied`.
inline void set_preset(diagnostic_preset p) noexcept {
    detail::g_preset.store(p, std::memory_order_relaxed);
    detail::g_level.store(detail::clamp_to_ceiling(detail::level_for(p)),
                          std::memory_order_relaxed);
}

/// @brief Last requested diagnostic preset.
[[nodiscard]] inline diagnostic_preset get_preset() noexcept {
    return detail::g_preset.load(std::memory_order_relaxed);
}

/// @brief Whether the requested preset is fully available in this build.
[[nodiscard]] inline bool preset_fully_applied() noexcept {
    return detail::level_for(get_preset()) == get_level();
}

/// @brief Scoped diagnostic preset, restored on exit.
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

/// @brief Raise a diagnostic error with the call site.
[[noreturn]] inline void panic(std::string_view category, std::string_view message,
                               std::source_location loc = std::source_location::current()) {
    std::string err = "[" + std::string(category) + "] Error at " + loc.file_name() + ":" +
                      std::to_string(loc.line()) + " in " + loc.function_name() + ":\n  " +
                      std::string(message);
    throw std::invalid_argument(err);
}

/// @brief Dimension check: `expected == actual`.
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

/// @brief Non-empty check: `size > 0`.
inline void check_non_empty(idx size, std::string_view label,
                            std::source_location loc = std::source_location::current()) {
    if constexpr (checks_compiled_in) {
        if (get_level() != diagnostic_level::off && size == 0) {
            panic("ValueError", std::string(label) + " cannot be empty (size is 0)", loc);
        }
    }
}

/// @brief Finiteness check: every value is finite.
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
