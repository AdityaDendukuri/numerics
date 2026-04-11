cmake_minimum_required(VERSION 3.20)
project(numerics VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(cmake/options.cmake)
include(cmake/sanitize.cmake)    # must come before any add_library/add_executable

include(cmake/cuda.cmake)       # sets NUMERICS_HAS_CUDA
include(cmake/mpi.cmake)        # sets NUMERICS_HAS_MPI

include(cmake/sources.cmake)    # defines LINALG_SOURCES

add_library(numerics ${LINALG_SOURCES})
target_include_directories(numerics PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

include(cmake/cuda.cmake)       # Phase 2: target properties
include(cmake/mpi.cmake)        # Phase 2: target properties
include(cmake/fftw.cmake)
include(cmake/blas.cmake)
include(cmake/openmp.cmake)
include(cmake/simd.cmake)
include(cmake/stdsimd.cmake)
include(cmake/apple.cmake)

include(cmake/physics.cmake)
include(cmake/install.cmake)

if(NUMERICS_BUILD_TESTS)
    include(cmake/dependencies.cmake)
    enable_testing()
    add_subdirectory(tests)
endif()

if(NUMERICS_BUILD_BENCHMARKS)
    if(NOT NUMERICS_BUILD_TESTS)
        include(cmake/dependencies.cmake)
    endif()
    add_subdirectory(benchmarks)
endif()

if(NUMERICS_BUILD_DOCS)
    add_subdirectory(docs)
endif()

if(NUMERICS_BUILD_REPORT)
    add_subdirectory(cmake/report)
endif()
