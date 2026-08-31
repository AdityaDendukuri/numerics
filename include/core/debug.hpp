/// @file core/debug.hpp
/// @brief Diagnostic presets, failure reporting, and dimension checks.
///
/// The scaffolding every module reports through. Sampling of mathematical
/// properties builds on this and lives in `algebra/debug.hpp`.
#pragma once

#include "core/types.hpp"
#include <algorithm>
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

enum class DiagnosticLevel {
    off = 0,
    basic = 1, ///< Dimension matching, non-empty, finite checks
    full = 2   ///< Includes sampled symmetry, adjoint, and positive-definiteness testing
};

/// @brief High-level execution and diagnostic presets for mathematical invariant enforcement.
enum class Preset : std::uint8_t {
    strict,     ///< Full runtime property sampling (x^T A x > 0, symmetry), loud warnings on untagged inputs.
    balanced,   ///< Dimension matching, structure bounds, warnings on untagged inputs (skips randomized sampling).
    unsafe,     ///< Prototyping mode: completely silences unverified invariant warnings and property checks.
    production, ///< Maximum throughput: all diagnostics disabled.
};

namespace preset {
inline constexpr Preset strict = Preset::strict;
inline constexpr Preset balanced = Preset::balanced;
inline constexpr Preset unsafe = Preset::unsafe;
inline constexpr Preset prototype = Preset::unsafe;
inline constexpr Preset production = Preset::production;
} // namespace preset

inline DiagnosticLevel g_level = DiagnosticLevel::full;

inline void set_level(DiagnosticLevel lvl) noexcept {
    g_level = lvl;
}

inline DiagnosticLevel get_level() noexcept {
    return g_level;
}

/// @brief Configure the global execution preset.
inline void set_preset(Preset p) noexcept {
    switch (p) {
    case Preset::strict:
        g_level = DiagnosticLevel::full;
        break;
    case Preset::balanced:
        g_level = DiagnosticLevel::basic;
        break;
    case Preset::unsafe:
    case Preset::production:
        g_level = DiagnosticLevel::off;
        break;
    }
}

/// @brief Get the current global execution preset.
inline Preset get_preset() noexcept {
    switch (g_level) {
    case DiagnosticLevel::full:
        return Preset::strict;
    case DiagnosticLevel::basic:
        return Preset::balanced;
    case DiagnosticLevel::off:
    default:
        return Preset::unsafe;
    }
}

/// @brief RAII guard to temporarily apply an execution preset within a scope.
class ScopedPreset {
  public:
    explicit ScopedPreset(Preset temp_preset) noexcept
        : previous_preset_(get_preset()) {
        set_preset(temp_preset);
    }

    ~ScopedPreset() noexcept {
        set_preset(previous_preset_);
    }

    ScopedPreset(const ScopedPreset &) = delete;
    ScopedPreset &operator=(const ScopedPreset &) = delete;
    ScopedPreset(ScopedPreset &&) = delete;
    ScopedPreset &operator=(ScopedPreset &&) = delete;

  private:
    Preset previous_preset_;
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
    if (g_level == DiagnosticLevel::off) {
        return;
    }
    if (expected != actual) {
        std::string msg = "Dimension mismatch for " + std::string(label) + ": expected " +
                          std::to_string(expected) + ", got " + std::to_string(actual);
        panic("DimensionError", msg, loc);
    }
}

/// @brief Verify a container is non-empty
inline void check_non_empty(idx size, std::string_view label,
                            std::source_location loc = std::source_location::current()) {
    if (g_level == DiagnosticLevel::off) {
        return;
    }
    if (size == 0) {
        panic("ValueError", std::string(label) + " cannot be empty (size is 0)", loc);
    }
}

/// @brief Verify all values in array are finite (not NaN or Inf)
template <typename T>
inline void check_finite(const T *data, idx n, std::string_view label,
                         std::source_location loc = std::source_location::current()) {
    if (g_level == DiagnosticLevel::off) {
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
} // namespace num::debug

namespace num {

using debug::get_preset;
using debug::Preset;
using debug::ScopedPreset;
using debug::set_preset;
namespace preset = debug::preset;

} // namespace num
