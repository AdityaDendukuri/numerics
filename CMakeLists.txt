cmake_minimum_required(VERSION 3.20)
project(numerics LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
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
include(cmake/fftw.cmake)    # before openmp: FFTW checks OPENMP_FOUND globally
include(cmake/blas.cmake)
include(cmake/openmp.cmake)
include(cmake/simd.cmake)
include(cmake/stdsimd.cmake)
include(cmake/apple.cmake)

include(cmake/physics.cmake)

include(cmake/dependencies.cmake)

if(NUMERICS_BUILD_FLUID_SIM)
    add_subdirectory(apps/fluid_sim)
endif()
if(NUMERICS_BUILD_FLUID_SIM_3D)
    add_subdirectory(apps/fluid_sim_3d)
endif()
if(NUMERICS_BUILD_EM_DEMO)
    add_subdirectory(apps/em_demo)
endif()
if(NUMERICS_BUILD_ISING)
    add_subdirectory(apps/ising)
endif()
if(NUMERICS_BUILD_NS_DEMO)
    add_subdirectory(apps/ns_demo)
endif()
if(NUMERICS_BUILD_TDSE)
    add_subdirectory(apps/tdse)
endif()
if(NUMERICS_BUILD_QUANTUM_DEMO)
    add_subdirectory(apps/quantum_demo)
endif()
if(NUMERICS_BUILD_NBODY)
    add_subdirectory(apps/nbody)
endif()
if(NUMERICS_BUILD_ISING_NUCLEATION)
    add_subdirectory(apps/ising_nucleation)
endif()

if(NUMERICS_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(NUMERICS_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

if(NUMERICS_BUILD_DOCS)
    add_subdirectory(docs)
endif()

if(NUMERICS_BUILD_REPORT)
    add_subdirectory(cmake/report)
endif()

if(NUMERICS_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()
