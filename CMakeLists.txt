cmake_minimum_required(VERSION 3.20)
project(numerics VERSION 1.0.0 LANGUAGES CXX)


set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(cmake/options.cmake)
include(cmake/sanitize.cmake)    # must come before any add_library/add_executable

include(cmake/cuda.cmake)       # Phase 1: CUDA detection
include(cmake/mpi.cmake)        # Phase 1: MPI detection

include(cmake/sources.cmake)    # defines NUMERICS_*_SOURCES

# --- Target Definitions ---

# 1. Tier-0 Raw Kernel (Header-only INTERFACE)
add_library(numerics_raw_kernel INTERFACE)
add_library(numerics::raw_kernel ALIAS numerics_raw_kernel)
target_include_directories(numerics_raw_kernel INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Backend configurations (attaches definitions and libraries to numerics_raw_kernel)
include(cmake/cuda.cmake)       # Phase 2: target properties
include(cmake/mpi.cmake)        # Phase 2: target properties
include(cmake/fftw.cmake)
include(cmake/blas.cmake)
include(cmake/openmp.cmake)
include(cmake/simd.cmake)
include(cmake/stdsimd.cmake)
include(cmake/apple.cmake)
include(cmake/suitesparse.cmake)

set(NUMERICS_HAS_BOOST_JSON FALSE)
if(NUMERICS_BUILD_IO)
    # Prefer Boost's package config; fall back for distributions that only ship FindBoost metadata.
    find_package(Boost CONFIG QUIET COMPONENTS json)
    if(NOT TARGET Boost::json)
        find_package(Boost QUIET COMPONENTS json)
    endif()
    if(TARGET Boost::json)
        set(NUMERICS_HAS_BOOST_JSON TRUE)
        message(STATUS "Boost.JSON: found; building numerics::io")
    else()
        message(STATUS "Boost.JSON: not found; optional numerics::io target disabled")
    endif()
endif()

# 2. Level 1 + Level 2 Kernel Target (Data structures, matrices, fields, operators; 0 solver overhead)
add_library(numerics_kernel ${NUMERICS_KERNEL_SOURCES})
add_library(numerics::kernel ALIAS numerics_kernel)
target_compile_features(numerics_kernel PUBLIC cxx_std_20)
target_include_directories(numerics_kernel PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(numerics_kernel PUBLIC numerics_raw_kernel)

# 3. Component Solver Targets (Layered Level 3 Targets)

# 3a. Linear Solvers, Factorizations & Eigensolvers (Depends on kernel)
add_library(numerics_solvers ${NUMERICS_CORE_SOURCES})
add_library(numerics::solvers ALIAS numerics_solvers)
target_compile_features(numerics_solvers PUBLIC cxx_std_20)
target_include_directories(numerics_solvers PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(numerics_solvers PUBLIC numerics_kernel)

# 3b. ODE Steppers & Integrators (Depends on solvers)
add_library(numerics_ode ${NUMERICS_ODE_SOURCES})
add_library(numerics::ode ALIAS numerics_ode)
target_compile_features(numerics_ode PUBLIC cxx_std_20)
target_include_directories(numerics_ode PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(numerics_ode PUBLIC numerics_solvers)

# 3c. PDE Field Solvers & Stencils (Depends on ode)
add_library(numerics_pde ${NUMERICS_PDE_SOURCES})
add_library(numerics::pde ALIAS numerics_pde)
target_compile_features(numerics_pde PUBLIC cxx_std_20)
target_include_directories(numerics_pde PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(numerics_pde PUBLIC numerics_ode)

# 3d. Spectral FFT Transforms (Depends on kernel)
add_library(numerics_spectral ${NUMERICS_SPECTRAL_SOURCES})
add_library(numerics::spectral ALIAS numerics_spectral)
target_compile_features(numerics_spectral PUBLIC cxx_std_20)
target_include_directories(numerics_spectral PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(numerics_spectral PUBLIC numerics_kernel)

# 4. Main Umbrella Target
add_library(numerics ${NUMERICS_SOURCES})
add_library(numerics::numerics ALIAS numerics)
target_compile_features(numerics PUBLIC cxx_std_20)
target_include_directories(numerics PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(numerics PUBLIC numerics_raw_kernel)

# Optional file-I/O layer. Core numerical users do not depend on JSON.
if(NUMERICS_HAS_BOOST_JSON)
    add_library(numerics_io ${NUMERICS_IO_SOURCES})
    add_library(numerics::io ALIAS numerics_io)
    target_compile_features(numerics_io PUBLIC cxx_std_20)
    target_include_directories(numerics_io PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    )
    target_link_libraries(numerics_io PUBLIC numerics::kernel Boost::json)
endif()

add_custom_target(tags
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tools/gen-tags.sh
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Generate editor tags"
)

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

if(NUMERICS_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

if(NUMERICS_BUILD_DOCS)
    add_subdirectory(docs)
endif()

if(NUMERICS_BUILD_REPORT)
    add_subdirectory(cmake/report)
endif()
