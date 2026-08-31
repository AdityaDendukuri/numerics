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

# Phase 2: Target configuration. MPI belongs only to numerics::mpi.
if(TARGET numerics_mpi AND NUMERICS_HAS_MPI)
    target_link_libraries(numerics_mpi PUBLIC MPI::MPI_CXX)
    target_compile_definitions(numerics_mpi PUBLIC NUMERICS_HAS_MPI)
endif()
