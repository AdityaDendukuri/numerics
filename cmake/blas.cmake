# BLAS — Backend::blas
# Searches for system BLAS and locates cblas.h.
# Falls back gracefully (blas -> blocked) if absent.
# Note: ACCELERATE_NEW_LAPACK is handled by cmake/apple.cmake.
if(NOT NUMERICS_USE_BLAS)
    return()
endif()

find_package(BLAS QUIET)
if(NOT BLAS_FOUND)
    message(STATUS "BLAS:  not found — Backend::blas falls back to blocked")
    message(STATUS "       Install: apt install libopenblas-dev  |  brew install openblas")
    return()
endif()

# Locate cblas.h — path varies across distros and macOS.
# PRIVATE: cblas.h is used only in .cpp files, not in installed public headers.
find_path(CBLAS_INCLUDE_DIR cblas.h
    PATH_SUFFIXES openblas blis
    HINTS
        /usr/include/openblas
        /usr/local/include/openblas
        /usr/include
        /usr/local/include
)

# Use BLAS::BLAS if the FindBLAS module created it (CMake >= 3.18), otherwise
# fall back to the raw library list.  Both are serialisable in NumericsTargets.cmake.
if(TARGET BLAS::BLAS)
    target_link_libraries(numerics PUBLIC BLAS::BLAS)
else()
    target_link_libraries(numerics PUBLIC ${BLAS_LIBRARIES})
endif()
target_compile_definitions(numerics PUBLIC NUMERICS_HAS_BLAS)

if(CBLAS_INCLUDE_DIR)
    # BUILD_INTERFACE: path is used during this build but not encoded in the
    # installed NumericsTargets.cmake.  Consumers who have BLAS installed will
    # have cblas.h on their own include path.
    target_include_directories(numerics PUBLIC
        $<BUILD_INTERFACE:${CBLAS_INCLUDE_DIR}>)
    message(STATUS "BLAS:  found  (${BLAS_LIBRARIES})")
    message(STATUS "cblas.h: ${CBLAS_INCLUDE_DIR}/cblas.h")
else()
    message(STATUS "BLAS:  found  (${BLAS_LIBRARIES})")
    message(STATUS "cblas.h: not found — add its directory to CMAKE_PREFIX_PATH")
endif()

# ── LAPACKE (C interface to LAPACK) ───────────────────────────────────────────
if(NOT NUMERICS_USE_LAPACK)
    return()
endif()

find_package(LAPACK QUIET)

find_path(LAPACKE_INCLUDE_DIR lapacke.h
    PATH_SUFFIXES openblas lapacke
    HINTS
        /usr/include/openblas
        /usr/local/include/openblas
        /usr/include
        /usr/local/include
)

# lapacke may be bundled in libopenblas or a separate liblapacke
find_library(LAPACKE_LIB NAMES lapacke openblas QUIET)

if(LAPACKE_INCLUDE_DIR AND (LAPACKE_LIB OR LAPACK_FOUND))
    target_compile_definitions(numerics PUBLIC NUMERICS_HAS_LAPACK)
    # PRIVATE: lapacke.h is used only in .cpp files, not in installed public headers.
    target_include_directories(numerics PRIVATE
        $<BUILD_INTERFACE:${LAPACKE_INCLUDE_DIR}>)
    if(LAPACKE_LIB)
        target_link_libraries(numerics PUBLIC ${LAPACKE_LIB})
    elseif(LAPACK_FOUND)
        if(TARGET LAPACK::LAPACK)
            target_link_libraries(numerics PUBLIC LAPACK::LAPACK)
        else()
            target_link_libraries(numerics PUBLIC ${LAPACK_LIBRARIES})
        endif()
    endif()
    set(NUMERICS_HAS_LAPACK ON CACHE INTERNAL "")
    message(STATUS "LAPACKE: found  (${LAPACKE_INCLUDE_DIR}/lapacke.h)")
else()
    message(STATUS "LAPACKE: not found — Backend::lapack falls back to default_backend")
    message(STATUS "         Install: apt install liblapacke-dev  |  brew install lapack")
    set(NUMERICS_HAS_LAPACK OFF CACHE INTERNAL "")
endif()
