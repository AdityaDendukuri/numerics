# External dependencies — FetchContent declarations
#
# Declarations are centralised here so tests and benchmarks can call
# FetchContent_MakeAvailable without repeating the URL and tag.
#
# FetchContent_Declare is idempotent on repeated calls with the
# same name, so subdirectories that call it defensively are safe.
include(FetchContent)

# Google Test — used by tests/
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.14.0
)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)

# Google Benchmark — used by benchmarks/
FetchContent_Declare(
    benchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG        v1.8.3
)
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
# Newer Clang releases diagnose constructs in older Google Benchmark headers
# (notably __COUNTER__ in a preprocessor expression) under -pedantic.  Third-party
# warnings must not make the numerics benchmark target unbuildable.
set(BENCHMARK_ENABLE_WERROR OFF CACHE BOOL "" FORCE)

# Raylib — only needed if an external app build defines one of these app options
# before including this dependency declaration file.
set(_any_app OFF)
foreach(_app NUMERICS_BUILD_APPS NUMERICS_BUILD_FLUID_SIM NUMERICS_BUILD_FLUID_SIM_3D
             NUMERICS_BUILD_EM_DEMO NUMERICS_BUILD_ISING NUMERICS_BUILD_NS_DEMO
             NUMERICS_BUILD_TDSE NUMERICS_BUILD_QUANTUM_DEMO)
    if(${_app})
        set(_any_app ON)
        break()
    endif()
endforeach()

if(_any_app)
    FetchContent_Declare(
        raylib
        GIT_REPOSITORY https://github.com/raysan5/raylib.git
        GIT_TAG        5.0
        GIT_SHALLOW    TRUE
    )
    set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(BUILD_GAMES    OFF CACHE BOOL "" FORCE)
endif()
