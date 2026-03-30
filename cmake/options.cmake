# Build options
option(NUMERICS_BUILD_TESTS         "Build unit tests"                   ON)
option(NUMERICS_BUILD_BENCHMARKS    "Build benchmarks"                   ON)
option(NUMERICS_BUILD_DOCS          "Build Doxygen documentation"        ON)
option(NUMERICS_ENABLE_CUDA         "Enable CUDA backend"                ON)
option(NUMERICS_ENABLE_MPI          "Enable MPI distributed backend"     ON)
option(NUMERICS_USE_BLAS            "Link BLAS/cblas for Backend::blas"  ON)
option(NUMERICS_USE_LAPACK          "Link LAPACKE for Backend::lapack"   ON)
option(NUMERICS_USE_OPENMP          "Enable OpenMP for Backend::omp"     ON)
option(NUMERICS_USE_FFTW            "Fetch and link FFTW3 for spectral/" ON)
option(NUMERICS_BUILD_REPORT        "Build markdown status report"       ON)
option(NUMERICS_BUILD_EXAMPLES      "Build example programs"             ON)

# Sanitizers — comma-separated list: asan, lsan, ubsan, tsan
# Example: -DNUMERICS_SANITIZE=asan,ubsan
# Use a separate build dir to avoid polluting the normal build.
set(NUMERICS_SANITIZE "" CACHE STRING "Comma-separated sanitizers to enable (asan|lsan|ubsan|tsan)")

# App targets — use umbrella or enable individually
option(NUMERICS_BUILD_APPS          "Build all raylib apps"              OFF)
option(NUMERICS_BUILD_FLUID_SIM     "Build 2D SPH fluid simulation"      OFF)
option(NUMERICS_BUILD_FLUID_SIM_3D  "Build 3D SPH fluid simulation"      OFF)
option(NUMERICS_BUILD_EM_DEMO       "Build electromagnetic field demo"   OFF)
option(NUMERICS_BUILD_ISING         "Build Ising model simulation"       OFF)
option(NUMERICS_BUILD_NS_DEMO       "Build 2D Navier-Stokes stress test" OFF)
option(NUMERICS_BUILD_TDSE          "Build 2D TDSE quantum simulation"   OFF)
option(NUMERICS_BUILD_QUANTUM_DEMO  "Build quantum circuit demo"          OFF)
option(NUMERICS_BUILD_NBODY              "Build gravitational N-body demo"         OFF)
option(NUMERICS_BUILD_ISING_NUCLEATION   "Build Ising nucleation (umbrella) demo"  OFF)

# Umbrella: NUMERICS_BUILD_APPS forces all app flags ON
if(NUMERICS_BUILD_APPS)
    set(NUMERICS_BUILD_FLUID_SIM         ON CACHE BOOL "" FORCE)
    set(NUMERICS_BUILD_FLUID_SIM_3D      ON CACHE BOOL "" FORCE)
    set(NUMERICS_BUILD_EM_DEMO           ON CACHE BOOL "" FORCE)
    set(NUMERICS_BUILD_ISING             ON CACHE BOOL "" FORCE)
    set(NUMERICS_BUILD_ISING_NUCLEATION  ON CACHE BOOL "" FORCE)
    set(NUMERICS_BUILD_NS_DEMO           ON CACHE BOOL "" FORCE)
    set(NUMERICS_BUILD_TDSE              ON CACHE BOOL "" FORCE)
    set(NUMERICS_BUILD_QUANTUM_DEMO      ON CACHE BOOL "" FORCE)
    set(NUMERICS_BUILD_NBODY             ON CACHE BOOL "" FORCE)
endif()
