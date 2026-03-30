# BLAS — Backend::blas
# Searches for system BLAS and locates cblas.h.
# Falls back gracefully (blas → blocked) if absent.
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

# Locate cblas.h — path varies across distros and macOS
find_path(CBLAS_INCLUDE_DIR cblas.h
    PATH_SUFFIXES openblas blis
    HINTS
        /usr/include/openblas
        /usr/local/include/openblas
        /usr/include
        /usr/local/include
)

target_link_libraries(numerics PUBLIC ${BLAS_LIBRARIES})
target_compile_definitions(numerics PUBLIC NUMERICS_HAS_BLAS)

if(CBLAS_INCLUDE_DIR)
    target_include_directories(numerics PUBLIC ${CBLAS_INCLUDE_DIR})
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
    target_include_directories(numerics PUBLIC ${LAPACKE_INCLUDE_DIR})
    if(LAPACKE_LIB)
        target_link_libraries(numerics PUBLIC ${LAPACKE_LIB})
    elseif(LAPACK_FOUND)
        target_link_libraries(numerics PUBLIC ${LAPACK_LIBRARIES})
    endif()
    set(NUMERICS_HAS_LAPACK ON CACHE INTERNAL "")
    message(STATUS "LAPACKE: found  (${LAPACKE_INCLUDE_DIR}/lapacke.h)")
else()
    message(STATUS "LAPACKE: not found — Backend::lapack falls back to default_backend")
    message(STATUS "         Install: apt install liblapacke-dev  |  brew install lapack")
    set(NUMERICS_HAS_LAPACK OFF CACHE INTERNAL "")
endif()
