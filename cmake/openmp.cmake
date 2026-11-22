# OpenMP — Backend::omp
if(NOT NUMERICS_USE_OPENMP)
    return()
endif()

# A Homebrew LAPACKE library may already load libomp. Use that same runtime instead of a
# compiler-bundled copy; loading both aborts when the first OpenMP region is entered.
if(APPLE AND LAPACKE_LIB AND EXISTS "${LAPACKE_LIB}")
    execute_process(COMMAND otool -L "${LAPACKE_LIB}"
        OUTPUT_VARIABLE _numerics_lapack_dependencies ERROR_QUIET)
    string(REGEX MATCH "[^ \t\n]*libomp\\.dylib" _numerics_lapack_openmp
        "${_numerics_lapack_dependencies}")
    if(_numerics_lapack_openmp AND EXISTS "${_numerics_lapack_openmp}")
        set(OpenMP_omp_LIBRARY "${_numerics_lapack_openmp}" CACHE FILEPATH
            "OpenMP runtime shared with LAPACKE" FORCE)
    endif()
endif()

find_package(OpenMP QUIET)
if(OpenMP_CXX_FOUND)
    target_link_libraries(numerics_backend_openmp INTERFACE OpenMP::OpenMP_CXX)
    target_compile_definitions(numerics_backend_openmp INTERFACE NUMERICS_HAS_OMP)
    message(STATUS "OpenMP: found  (${OpenMP_CXX_VERSION})")
else()
    message(STATUS "OpenMP: not found — Backend::omp falls back to seq")
endif()
