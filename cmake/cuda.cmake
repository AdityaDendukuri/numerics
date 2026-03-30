# CUDA — detection (Phase 1) and target configuration (Phase 2)
#
# This file is included twice from the top-level CMakeLists.txt:
#   1. Before add_library — detects CUDA, sets NUMERICS_HAS_CUDA
#   2. After add_library  — applies target properties
#
# The if(NOT DEFINED ...) guard ensures detection runs only once.
# The if(TARGET numerics ...) guard triggers target config only
# on the second inclusion.

# Phase 1: Detection
if(NOT DEFINED NUMERICS_HAS_CUDA)
    set(NUMERICS_HAS_CUDA FALSE)
    if(NUMERICS_ENABLE_CUDA)
        include(CheckLanguage)
        check_language(CUDA)
        if(CMAKE_CUDA_COMPILER)
            enable_language(CUDA)
            set(CMAKE_CUDA_STANDARD 17)
            find_package(CUDAToolkit QUIET)
            if(CUDAToolkit_FOUND)
                set(NUMERICS_HAS_CUDA TRUE)
            endif()
        endif()
    endif()
    message(STATUS "CUDA support:  ${NUMERICS_HAS_CUDA}")
endif()

# Phase 2: Target configuration (only after add_library)
if(TARGET numerics AND NUMERICS_HAS_CUDA)
    target_link_libraries(numerics PUBLIC CUDA::cudart)
    target_compile_definitions(numerics PUBLIC NUMERICS_HAS_CUDA)
    set_target_properties(numerics PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE  ON
    )
endif()
