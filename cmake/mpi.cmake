# MPI — detection (Phase 1) and target configuration (Phase 2)
#
# Same two-phase pattern as cuda.cmake — include before
# add_library for detection, and again after for target config.

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

# Phase 2: Target configuration (only after add_library)
if(TARGET numerics AND NUMERICS_HAS_MPI)
    target_link_libraries(numerics PUBLIC MPI::MPI_CXX)
    target_compile_definitions(numerics PUBLIC NUMERICS_HAS_MPI)
endif()
