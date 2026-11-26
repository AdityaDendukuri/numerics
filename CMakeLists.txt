cmake_minimum_required(VERSION 3.20)
project(numerics VERSION 1.2.0 LANGUAGES CXX)


set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Symlink compile_commands.json to the source root for clangd and other editor tooling.
# Only for a top-level build, and never fatal: package managers configure several build
# trees from one source tree at once, and a subproject must not touch its parent's tree.
if(CMAKE_EXPORT_COMPILE_COMMANDS AND PROJECT_IS_TOP_LEVEL)
    file(CREATE_LINK
        "${CMAKE_BINARY_DIR}/compile_commands.json"
        "${CMAKE_CURRENT_SOURCE_DIR}/compile_commands.json"
        RESULT _numerics_link_result
        SYMBOLIC
    )
endif()

include(cmake/options.cmake)
include(cmake/sanitize.cmake)    # must come before any add_library/add_executable

include(cmake/cuda.cmake)       # Phase 1: CUDA detection
include(cmake/mpi.cmake)        # Phase 1: MPI detection

include(cmake/sources.cmake)    # defines NUMERICS_*_SOURCES

# --- Target Definitions ---
#
# The library is header-only. Everything below is an INTERFACE target that
# carries include paths and, for a backend, a compile definition and the external
# library to link. Two compiled targets exist and both are optional.

# 1. Foundational targets. These stay dependency-free even when every optional
#    backend is present on the configuring machine.
add_library(numerics_kernel INTERFACE)
add_library(numerics::kernel ALIAS numerics_kernel)
set_target_properties(numerics_kernel PROPERTIES EXPORT_NAME kernel)
target_include_directories(numerics_kernel INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    # Installed headers live under <prefix>/include/numerics so they never collide
    # with other packages. Exposing that directory keeps <numerics.hpp> and the
    # library's own "core/..." includes working; <prefix>/include also allows
    # <numerics/numerics.hpp>.
    $<INSTALL_INTERFACE:include/numerics>
    $<INSTALL_INTERFACE:include>
)
target_compile_features(numerics_kernel INTERFACE cxx_std_20)

add_library(numerics_core INTERFACE)
add_library(numerics::core ALIAS numerics_core)
set_target_properties(numerics_core PROPERTIES EXPORT_NAME core)
target_link_libraries(numerics_core INTERFACE numerics_kernel)

# 2. Optional capabilities. Detection modules configure only these targets; the
#    kernel and core above never receive vendor headers, libraries, definitions,
#    or architecture flags.
foreach(_backend blas lapack openmp simd)
    add_library(numerics_backend_${_backend} INTERFACE)
    add_library(numerics::${_backend} ALIAS numerics_backend_${_backend})
    add_library(numerics::backend::${_backend} ALIAS numerics_backend_${_backend})
    set_target_properties(numerics_backend_${_backend} PROPERTIES EXPORT_NAME ${_backend})
    target_link_libraries(numerics_backend_${_backend} INTERFACE numerics_core)
endforeach()

add_library(numerics_backends INTERFACE)
add_library(numerics::backends ALIAS numerics_backends)
set_target_properties(numerics_backends PROPERTIES EXPORT_NAME backends)
target_link_libraries(numerics_backends INTERFACE
    numerics_backend_blas
    numerics_backend_lapack
    numerics_backend_openmp
    numerics_backend_simd
)

include(cmake/cuda.cmake)
include(cmake/mpi.cmake)
include(cmake/blas.cmake)
include(cmake/openmp.cmake)
include(cmake/simd.cmake)
include(cmake/stdsimd.cmake)
include(cmake/apple.cmake)

# 3. The umbrella. Header-only: everything else in this file is optional.
add_library(numerics INTERFACE)
add_library(numerics::numerics ALIAS numerics)
set_target_properties(numerics PROPERTIES EXPORT_NAME numerics)
target_link_libraries(numerics INTERFACE numerics_core numerics_backends)
target_compile_features(numerics INTERFACE cxx_std_20)

# Storage alignment must reach consumers: it is an argument to the aligned operator
# delete, so every translation unit must agree on it.
target_compile_definitions(numerics INTERFACE
    NUMERICS_STORAGE_ALIGNMENT=${NUMERICS_STORAGE_ALIGNMENT}
)

# 4. Optional compiled capabilities.

# Distributed-memory bindings.
if(NUMERICS_HAS_MPI)
    add_library(numerics_mpi ${NUMERICS_MPI_SOURCES})
    target_link_libraries(numerics_mpi PUBLIC numerics)
else()
    add_library(numerics_mpi INTERFACE)
    target_link_libraries(numerics_mpi INTERFACE numerics)
endif()
add_library(numerics::mpi ALIAS numerics_mpi)
set_target_properties(numerics_mpi PROPERTIES EXPORT_NAME mpi)

if(NUMERICS_HAS_CUDA)
    add_library(numerics_cuda ${NUMERICS_CUDA_SOURCES})
    add_library(numerics::cuda ALIAS numerics_cuda)
    set_target_properties(numerics_cuda PROPERTIES EXPORT_NAME cuda)
    target_link_libraries(numerics_cuda PUBLIC numerics)
endif()

# 4. Compatibility aliases for the previous per-domain targets. All header-only
#    now, so each simply forwards to numerics.
foreach(_alias solvers ode pde spectral plot)
    add_library(numerics_${_alias} INTERFACE)
    add_library(numerics::${_alias} ALIAS numerics_${_alias})
    set_target_properties(numerics_${_alias} PROPERTIES EXPORT_NAME ${_alias})
    target_link_libraries(numerics_${_alias} INTERFACE numerics)
endforeach()

add_custom_target(tags
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tools/gen-tags.sh
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Generate editor tags"
)

# Phase 2 of the toolchain modules: attach settings to numerics' own targets now
# that they exist. The earlier includes above run before they are created.
include(cmake/apple.cmake)
include(cmake/cuda.cmake)
include(cmake/mpi.cmake)

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

# The docs carry ~1500 lines of example code. Nothing compiled it until this target
# existed, so an API rename could leave the documentation describing a library that no
# longer exists. Not wired into ctest: it compiles the umbrella header once per block and
# takes minutes, where the whole suite takes seconds.
find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_Interpreter_FOUND)
    add_custom_target(check-docs
        COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/cmake/check_doc_examples.py
                --source-dir ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Compiling every C++ example in the docs pages"
        VERBATIM)
endif()
