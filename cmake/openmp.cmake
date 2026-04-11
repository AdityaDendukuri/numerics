# OpenMP — Backend::omp
# Falls back gracefully (omp → seq) if absent.
if(NOT NUMERICS_USE_OPENMP)
    return()
endif()

find_package(OpenMP QUIET)
if(OpenMP_CXX_FOUND)
    target_link_libraries(numerics PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(numerics PUBLIC NUMERICS_HAS_OMP)
    message(STATUS "OpenMP: found  (${OpenMP_CXX_VERSION})")
else()
    message(STATUS "OpenMP: not found — Backend::omp falls back to seq")
endif()
