# Build options for numerics
if(PROJECT_IS_TOP_LEVEL)
    option(NUMERICS_BUILD_TESTS      "Build unit tests"               OFF)
    option(NUMERICS_BUILD_BENCHMARKS "Build benchmarks"               OFF)
    option(NUMERICS_BUILD_EXAMPLES   "Build example targets"          ON)
    option(NUMERICS_BUILD_DOCS       "Build Doxygen documentation"    OFF)
    option(NUMERICS_BUILD_REPORT     "Build benchmark report"         OFF)
else()
    option(NUMERICS_BUILD_TESTS      "Build unit tests"               OFF)
    option(NUMERICS_BUILD_BENCHMARKS "Build benchmarks"               OFF)
    option(NUMERICS_BUILD_EXAMPLES   "Build example targets"          OFF)
    option(NUMERICS_BUILD_DOCS       "Build Doxygen documentation"    OFF)
    option(NUMERICS_BUILD_REPORT     "Build benchmark report"         OFF)
endif()



option(NUMERICS_ENABLE_CUDA  "Enable CUDA backend"                ON)
option(NUMERICS_ENABLE_MPI   "Enable MPI distributed backend"     ON)
option(NUMERICS_USE_BLAS     "Link BLAS/cblas for Backend::blas"  ON)
option(NUMERICS_USE_LAPACK   "Link LAPACKE for Backend::lapack"   ON)
option(NUMERICS_USE_OPENMP   "Enable OpenMP for Backend::omp"     ON)
option(NUMERICS_USE_FFTW     "Link FFTW3 for spectral/"           ON)
option(NUMERICS_USE_SUITESPARSE "Enable optional SuiteSparse KLU sparse-direct backend" ON)
option(NUMERICS_BUILD_IO "Build the optional file-I/O target when available" ON)

# Sanitizers — comma-separated list: asan, lsan, ubsan, tsan
set(NUMERICS_SANITIZE "" CACHE STRING "Comma-separated sanitizers to enable (asan|lsan|ubsan|tsan)")
