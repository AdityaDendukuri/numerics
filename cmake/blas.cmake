# BLAS — Backend::blas
# Searches for system BLAS and locates cblas.h.
# Falls back gracefully (blas -> blocked) if absent.
if(NOT NUMERICS_USE_BLAS)
    return()
endif()

find_package(BLAS QUIET)
if(NOT BLAS_FOUND)
    message(STATUS "BLAS:  not found — Backend::blas falls back to blocked")
    message(STATUS "       Install: apt install libopenblas-dev  |  brew install openblas")
    return()
endif()

find_path(CBLAS_INCLUDE_DIR cblas.h
    PATH_SUFFIXES openblas blis
    HINTS
        /usr/include/openblas
        /usr/local/include/openblas
        /usr/include
        /usr/local/include
)

if(TARGET BLAS::BLAS)
    target_link_libraries(numerics_backend_blas INTERFACE BLAS::BLAS)
else()
    target_link_libraries(numerics_backend_blas INTERFACE ${BLAS_LIBRARIES})
endif()
target_compile_definitions(numerics_backend_blas INTERFACE NUMERICS_HAS_BLAS)

if(CBLAS_INCLUDE_DIR)
    target_include_directories(numerics_backend_blas INTERFACE ${CBLAS_INCLUDE_DIR})
    message(STATUS "BLAS:  found  (${BLAS_LIBRARIES})")
    message(STATUS "cblas.h: ${CBLAS_INCLUDE_DIR}/cblas.h")
else()
    message(STATUS "BLAS:  found  (${BLAS_LIBRARIES})")
    message(STATUS "cblas.h: not found — add its directory to CMAKE_PREFIX_PATH")
endif()

# ── LAPACKE (Standard C interface to LAPACK across Linux & macOS) ──────────────
if(NOT NUMERICS_USE_LAPACK)
    return()
endif()

find_package(LAPACK QUIET)

# An optimized LAPACKE first (OpenBLAS bundles one), the reference package last.
# Reference LAPACK on top of an optimized BLAS is fine -- its factorizations are
# blocked over dgemm/dtrsm -- but the Homebrew and Debian `lapack` packages link
# their own reference BLAS, and that combination runs several times slower than
# numerics' own kernel.
find_path(LAPACKE_INCLUDE_DIR lapacke.h
    PATH_SUFFIXES openblas lapacke
    HINTS
        /opt/homebrew/opt/openblas/include
        /usr/local/opt/openblas/include
        /usr/include/openblas
        /usr/local/include/openblas
        /opt/homebrew/opt/lapack/include
        /usr/local/opt/lapack/include
        /usr/include
        /usr/local/include
)

find_library(LAPACKE_LIB NAMES openblas lapacke
    HINTS
        /opt/homebrew/opt/openblas/lib
        /usr/local/opt/openblas/lib
        /opt/homebrew/opt/lapack/lib
        /usr/local/opt/lapack/lib
    QUIET)

# Is the LAPACK that was found backed by an optimized BLAS? Checked by reading
# the shared-library dependencies of the LAPACKE library and of any liblapack it
# pulls in, and looking for a known optimized implementation. Overridable:
# NUMERICS_LAPACK_OPTIMIZED=ON/OFF skips the check.
set(NUMERICS_LAPACK_OPTIMIZED "AUTO" CACHE STRING
    "Whether the found LAPACK is backed by an optimized BLAS (AUTO, ON, OFF)")
set(_fast_blas_pattern "openblas|mkl|blis|flexiblas|armpl|atlas|Accelerate|nvpl|essl|libsci")

function(_numerics_lapack_is_optimized lib out_var)
    if(lib MATCHES "${_fast_blas_pattern}")
        set(${out_var} ON PARENT_SCOPE)
        return()
    endif()
    set(_deps "")
    if(APPLE)
        execute_process(COMMAND otool -L "${lib}" OUTPUT_VARIABLE _deps ERROR_QUIET)
    elseif(UNIX)
        execute_process(COMMAND ldd "${lib}" OUTPUT_VARIABLE _deps ERROR_QUIET)
    endif()
    if(_deps MATCHES "${_fast_blas_pattern}")
        set(${out_var} ON PARENT_SCOPE)
        return()
    endif()
    # One level down: liblapacke -> liblapack -> libblas.
    string(REGEX MATCHALL "[^ \t\n]*liblapack[^ \t\n]*" _lapack_deps "${_deps}")
    get_filename_component(_lib_dir "${lib}" DIRECTORY)
    foreach(_dep IN LISTS _lapack_deps)
        get_filename_component(_dep_name "${_dep}" NAME)
        set(_candidate "${_lib_dir}/${_dep_name}")
        if(NOT EXISTS "${_candidate}")
            set(_candidate "${_dep}")
        endif()
        if(EXISTS "${_candidate}")
            if(APPLE)
                execute_process(COMMAND otool -L "${_candidate}" OUTPUT_VARIABLE _sub ERROR_QUIET)
            else()
                execute_process(COMMAND ldd "${_candidate}" OUTPUT_VARIABLE _sub ERROR_QUIET)
            endif()
            if(_sub MATCHES "${_fast_blas_pattern}")
                set(${out_var} ON PARENT_SCOPE)
                return()
            endif()
        endif()
    endforeach()
    set(${out_var} OFF PARENT_SCOPE)
endfunction()

if(LAPACKE_INCLUDE_DIR AND (LAPACKE_LIB OR LAPACK_FOUND))
    target_compile_definitions(numerics_backend_lapack INTERFACE NUMERICS_HAS_LAPACK)
    target_include_directories(numerics_backend_lapack INTERFACE ${LAPACKE_INCLUDE_DIR})
    if(LAPACKE_LIB)
        target_link_libraries(numerics_backend_lapack INTERFACE ${LAPACKE_LIB})
        set(_lapack_probe "${LAPACKE_LIB}")
    elseif(LAPACK_FOUND)
        if(TARGET LAPACK::LAPACK)
            target_link_libraries(numerics_backend_lapack INTERFACE LAPACK::LAPACK)
        else()
            target_link_libraries(numerics_backend_lapack INTERFACE ${LAPACK_LIBRARIES})
        endif()
        list(GET LAPACK_LIBRARIES 0 _lapack_probe)
    endif()
    set(NUMERICS_HAS_LAPACK ON CACHE INTERNAL "")

    if(NUMERICS_LAPACK_OPTIMIZED STREQUAL "AUTO")
        _numerics_lapack_is_optimized("${_lapack_probe}" _lapack_fast)
    else()
        set(_lapack_fast ${NUMERICS_LAPACK_OPTIMIZED})
    endif()
    if(_lapack_fast)
        message(STATUS "LAPACKE: found  (${_lapack_probe}), optimized BLAS underneath")
    else()
        # The bindings stay available under num::lapack::*, but the default
        # dense factorizations use the kernel, which is faster than reference
        # LAPACK on reference BLAS.
        target_compile_definitions(numerics_backend_lapack INTERFACE NUMERICS_LAPACK_REFERENCE)
        message(STATUS "LAPACKE: found  (${_lapack_probe}), reference BLAS underneath -- "
                       "dense factorizations default to the kernel; "
                       "install OpenBLAS or set NUMERICS_LAPACK_OPTIMIZED=ON")
    endif()
else()
    message(STATUS "LAPACKE: not found — Backend::lapack falls back to default_backend")
    message(STATUS "         Install: apt install liblapacke-dev  |  brew install openblas")
    set(NUMERICS_HAS_LAPACK OFF CACHE INTERNAL "")
endif()
