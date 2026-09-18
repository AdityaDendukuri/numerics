/// @file core/call_site.hpp
/// @brief The caller's source location, captured correctly by every compiler.
#pragma once

#include <source_location>

namespace num {

/// @brief A source location that records where a function was called.
///
/// `std::source_location::current()` as a default argument of a *function
/// template* records the wrong place under GCC (through 14): the default
/// argument is instantiated once per specialization, and a program with many
/// callers of `assume<spd>(mat)` sees every proof carry the first caller's
/// location, from whichever translation unit the linker kept. The same call
/// through a default argument of a non-template constructor is evaluated at
/// each call, on every compiler. So the templates take a `call_site` whose
/// default is `{}`, and the constructor does the capturing.
struct call_site {
    std::source_location location;

    // NOLINTNEXTLINE(google-explicit-constructor): the implicit conversion is the point.
    constexpr call_site(std::source_location where = std::source_location::current()) noexcept
        : location(where) {}
};

} // namespace num
