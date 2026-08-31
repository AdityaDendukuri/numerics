cmake_minimum_required(VERSION 3.20)
project(numerics VERSION 1.0.0 LANGUAGES CXX)


set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Symlink compile_commands.json to project root for clangd / Neovim LSP
if(CMAKE_EXPORT_COMPILE_COMMANDS)
    file(CREATE_LINK
        "${CMAKE_BINARY_DIR}/compile_commands.json"
        "${CMAKE_CURRENT_SOURCE_DIR}/compile_commands.json"
        SYMBOLIC COPY_ON_ERROR
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
    $<INSTALL_INTERFACE:include>
)
target_compile_features(numerics_kernel INTERFACE cxx_std_20)

add_library(numerics_core INTERFACE)
add_library(numerics::core ALIAS numerics_core)
set_target_properties(numerics_core PROPERTIES EXPORT_NAME core)
target_link_libraries(numerics_core INTERFACE numerics_kernel)

# Kept for source compatibility with builds that named the old target.
add_library(numerics_raw_kernel INTERFACE)
add_library(numerics::raw_kernel ALIAS numerics_raw_kernel)
set_target_properties(numerics_raw_kernel PROPERTIES EXPORT_NAME raw_kernel)
target_link_libraries(numerics_raw_kernel INTERFACE numerics_kernel)

# 2. Optional capabilities. Detection modules configure only these targets; the
#    kernel and core above never receive vendor headers, libraries, definitions,
#    or architecture flags.
foreach(_backend blas lapack openmp fftw suitesparse simd)
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
    numerics_backend_fftw
    numerics_backend_suitesparse
    numerics_backend_simd
)

include(cmake/cuda.cmake)
include(cmake/mpi.cmake)
include(cmake/fftw.cmake)
include(cmake/blas.cmake)
include(cmake/openmp.cmake)
include(cmake/simd.cmake)
include(cmake/stdsimd.cmake)
include(cmake/apple.cmake)
include(cmake/suitesparse.cmake)

if(NUMERICS_BUILD_IO)
    include(FetchContent)
    FetchContent_Declare(
        nlohmann_json
        URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    FetchContent_MakeAvailable(nlohmann_json)
endif()

# 3. The compatibility umbrella. Mostly headers; what remains compiled is listed in
#    cmake/sources.cmake and is either an external binding or a translation unit
#    whose file-local helpers are not worth inlining.
add_library(numerics ${NUMERICS_CORE_SOURCES} ${NUMERICS_INSTANTIATION_SOURCES})
add_library(numerics::numerics ALIAS numerics)
set_target_properties(numerics PROPERTIES EXPORT_NAME numerics)
target_link_libraries(numerics PUBLIC numerics_core numerics_backends)
target_compile_features(numerics PUBLIC cxx_std_20)

# The numerics build shares one instantiation of the common container types
# across its translation units. Deliberately not part of the public interface:
# headers copied out of this tree must instantiate implicitly.
target_compile_definitions(numerics PUBLIC
    $<BUILD_INTERFACE:NUMERICS_EXTERN_TEMPLATES>
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
    target_link_libraries(numerics_${_alias} INTERFACE numerics)
endforeach()

if(NUMERICS_BUILD_IO)
    add_library(numerics_io INTERFACE)
    add_library(numerics::io ALIAS numerics_io)
    target_link_libraries(numerics_io INTERFACE numerics
        $<BUILD_INTERFACE:nlohmann_json::nlohmann_json>)
endif()

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
