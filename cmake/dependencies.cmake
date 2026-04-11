# External dependencies — FetchContent declarations
#
# Declarations are centralised here so every subdirectory
# (tests/, benchmarks/, apps/*) can call FetchContent_MakeAvailable
# without repeating the URL and tag.
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

# Google Benchmark — used by benchmarks/
FetchContent_Declare(
    benchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG        v1.8.3
)
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)

# Raylib — only needed when at least one app target is enabled
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
