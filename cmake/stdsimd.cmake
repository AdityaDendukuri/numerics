# std::experimental::simd detection (GCC 11+ / libstdc++)
# Sets NUMERICS_HAS_STD_SIMD and the matching compile definition on the
# numerics target when <experimental/simd> is available and usable.
include(CheckCXXSourceCompiles)

# Save/restore flags: the test must compile at the project standard (C++17).
set(_saved_required ${CMAKE_CXX_STANDARD_REQUIRED})
set(_saved_std      ${CMAKE_CXX_STANDARD})
set(CMAKE_CXX_STANDARD          17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

check_cxx_source_compiles("
#include <experimental/simd>
namespace stdx = std::experimental;
int main() {
    using vd = stdx::simd<double, stdx::simd_abi::native<double>>;
    constexpr int W = static_cast<int>(vd::size());
    (void)W;
    vd a([](int i) { return static_cast<double>(i); });
    vd b = a * a;
    double s = 0;
    for (int i = 0; i < W; ++i) s += b[i];
    return s < 0 ? 1 : 0;
}
" NUMERICS_HAS_STD_SIMD)

set(CMAKE_CXX_STANDARD          ${_saved_std})
set(CMAKE_CXX_STANDARD_REQUIRED ${_saved_required})

if(NUMERICS_HAS_STD_SIMD)
    target_compile_definitions(numerics PUBLIC NUMERICS_HAS_STD_SIMD)
    message(STATUS "std::simd: <experimental/simd> available")
else()
    message(STATUS "std::simd: not available (needs GCC 11+ with libstdc++)")
endif()
