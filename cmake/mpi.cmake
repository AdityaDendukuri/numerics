# MPI — detection (Phase 1) and target configuration (Phase 2)

# Phase 1: Detection
if(NOT DEFINED NUMERICS_HAS_MPI)
    set(NUMERICS_HAS_MPI FALSE)
    if(NUMERICS_ENABLE_MPI)
        find_package(MPI QUIET)
        if(MPI_FOUND)
            set(NUMERICS_HAS_MPI TRUE)
        endif()
    endif()
    message(STATUS "MPI support:   ${NUMERICS_HAS_MPI}")
endif()

# Phase 2: Target configuration
if(TARGET numerics_raw_kernel AND NUMERICS_HAS_MPI)
    target_link_libraries(numerics_raw_kernel INTERFACE MPI::MPI_CXX)
    target_compile_definitions(numerics_raw_kernel INTERFACE NUMERICS_HAS_MPI)
endif()
