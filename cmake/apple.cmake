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
target_include_directories(numerics PUBLIC ${Boost_INCLUDE_DIRS})
target_compile_definitions(numerics PUBLIC NUMERICS_USE_BOOST_MATH)

# macOS 13.3+: opt into the new Accelerate/LAPACK headers to
# silence deprecation warnings when BLAS is the Accelerate framework
target_compile_definitions(numerics PUBLIC ACCELERATE_NEW_LAPACK)

message(STATUS "Apple: Boost.Math for C++17 special functions (${Boost_INCLUDE_DIRS})")
