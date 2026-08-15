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

find_package(Boost REQUIRED)
target_include_directories(numerics_raw_kernel INTERFACE ${Boost_INCLUDE_DIRS})
target_compile_definitions(numerics_raw_kernel INTERFACE NUMERICS_USE_BOOST_MATH ACCELERATE_NEW_LAPACK)

message(STATUS "Apple: Boost.Math for C++17 special functions (${Boost_INCLUDE_DIRS})")
