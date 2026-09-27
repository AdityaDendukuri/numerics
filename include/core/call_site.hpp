/// @file core/call_site.hpp
/// @brief The caller's source location, captured correctly by every compiler.
#pragma once

#include <source_location>

namespace num {

/// @brief A source location that records where a function was called.
///
/// Under GCC through 14, `std::source_location::current()` as a default argument of a function
/// template records one location per specialization. A non-template constructor's default is
/// evaluated at each call, so the templates take a `call_site` defaulted to `{}`.
struct call_site {
    std::source_location location;

    // NOLINTNEXTLINE(google-explicit-constructor): the implicit conversion is the point.
    constexpr call_site(std::source_location where = std::source_location::current()) noexcept
        : location(where) {}
};

} // namespace num
