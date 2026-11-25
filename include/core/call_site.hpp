/// @file core/call_site.hpp
/// @brief Caller source location, captured portably.
#pragma once

#include <source_location>

namespace num {

/// @brief Source location of the call site.
/// Pass as a defaulted `{}` argument; GCC <= 14 misrecords it in function templates.
struct call_site {
    std::source_location location;

    // NOLINTNEXTLINE(google-explicit-constructor): the implicit conversion is the point.
    constexpr call_site(std::source_location where = std::source_location::current()) noexcept
        : location(where) {}
};

} // namespace num
