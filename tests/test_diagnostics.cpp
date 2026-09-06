/// @file test_diagnostics.cpp
/// @brief The diagnostic ceiling and the runtime preset are separate dials.
///
/// `NUMERICS_DIAGNOSTICS` decides what checking code exists in a build. The runtime preset
/// decides whether it runs. Conflating the two produced three defects that these tests
/// pin down: `preset::production` did not remove any code, `get_preset()` could not report
/// `production` back, and the default was the most expensive setting in every build.
///
/// This file is compiled at ceiling 2 along with the rest of the suite, so the tests that
/// need sampling have it. The ones about clamping check the constants rather than the
/// build they happen to run under.

#include "core/debug.hpp"
#include "linear/matrix_properties.hpp"
#include "operator/dense.hpp"
#include "operator/properties.hpp"
#include <gtest/gtest.h>

using namespace num;

namespace {

/// Restores whatever preset was in force, so ordering between tests cannot matter.
struct preset_restorer {
    diagnostic_preset saved = get_preset();
    ~preset_restorer() { set_preset(saved); }
};

} // namespace

TEST(Diagnostics, EveryPresetRoundTrips) {
    preset_restorer restore;
    // `unsafe` and `production` share a diagnostic level, so a `get_preset` that derived
    // the preset from the level reported `unsafe` for both.
    for (const auto p : {preset::strict, preset::balanced, preset::unsafe, preset::production}) {
        set_preset(p);
        EXPECT_EQ(get_preset(), p);
    }
}

TEST(Diagnostics, RequestsAreClampedToTheCompileTimeCeiling) {
    preset_restorer restore;
    set_preset(preset::strict);

    if constexpr (debug::sampling_compiled_in) {
        EXPECT_EQ(debug::get_level(), debug::diagnostic_level::full);
        EXPECT_TRUE(preset_fully_applied());
    } else {
        // The request is remembered, but the level cannot exceed what was compiled.
        EXPECT_EQ(get_preset(), diagnostic_preset::strict);
        EXPECT_EQ(debug::get_level(), debug::compiled_level);
        EXPECT_FALSE(preset_fully_applied())
            << "asking for more checking than the build contains must be reported, "
               "not silently downgraded";
    }
}

TEST(Diagnostics, LoweringThePresetIsAlwaysHonoured) {
    preset_restorer restore;
    // Clamping only ever reduces, so every build can turn diagnostics off.
    set_preset(preset::production);
    EXPECT_EQ(debug::get_level(), debug::diagnostic_level::off);
    EXPECT_TRUE(preset_fully_applied());
}

TEST(Diagnostics, TheCeilingAgreesWithItsConstants) {
    EXPECT_EQ(debug::checks_compiled_in, NUMERICS_DIAGNOSTICS >= 1);
    EXPECT_EQ(debug::sampling_compiled_in, NUMERICS_DIAGNOSTICS >= 2);
    EXPECT_EQ(static_cast<int>(debug::compiled_level), NUMERICS_DIAGNOSTICS);
}

TEST(Diagnostics, SamplingRunsWhenCompiledInAndThePresetAsksForIt) {
    if constexpr (!debug::sampling_compiled_in) {
        GTEST_SKIP() << "built below the sampling ceiling";
    } else {
        preset_restorer restore;
        set_preset(preset::strict);

        mat indefinite(2, 2, 0.0);
        indefinite(0, 0) = -5.0;
        indefinite(1, 1) = 1.0;
        operators::dense_op op(indefinite);

        EXPECT_THROW(static_cast<void>(operators::assume_spd(op)), std::invalid_argument);

        // The same claim under `production` is a tag and nothing more.
        set_preset(preset::production);
        EXPECT_NO_THROW(static_cast<void>(operators::assume_spd(op)));
    }
}

TEST(Diagnostics, ScopedPresetRestoresTheRequestedPresetNotTheLevel) {
    preset_restorer restore;
    set_preset(preset::production);
    {
        const scoped_preset guard(preset::strict);
        EXPECT_EQ(get_preset(), diagnostic_preset::strict);
    }
    // Restoring through `get_preset` is why the preset has to be stored: recovering it
    // from the level would have brought back `unsafe` here.
    EXPECT_EQ(get_preset(), diagnostic_preset::production);
}
