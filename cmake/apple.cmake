# Apple platform workarounds
#
# Apple libc++ does not implement C++17 special math functions:
# Bessel functions, elliptic integrals, orthogonal polynomials,
# expint, zeta, beta — <cmath> declares them but they are not
# linked. Boost.Math provides transparent drop-in implementations.
#
# Install: brew install boost
if(NOT APPLE)
    return()
endif()

if(POLICY CMP0167)
    cmake_policy(SET CMP0167 NEW)
endif()

find_package(Boost REQUIRED)
target_include_directories(numerics_raw_kernel INTERFACE ${Boost_INCLUDE_DIRS})
target_compile_definitions(numerics_raw_kernel INTERFACE NUMERICS_USE_BOOST_MATH ACCELERATE_NEW_LAPACK)
# Accelerate keeps the legacy CBLAS entry points for source compatibility but
# marks them deprecated even when they are the supported ABI on this platform.
# Keep that SDK warning out of Numerics diagnostics; clang-tidy still checks
# the calling code itself.
target_compile_options(numerics_raw_kernel INTERFACE
    $<$<COMPILE_LANG_AND_ID:CXX,AppleClang,Clang>:-Wno-deprecated-declarations>)

message(STATUS "Apple: Boost.Math for C++17 special functions (${Boost_INCLUDE_DIRS})")
