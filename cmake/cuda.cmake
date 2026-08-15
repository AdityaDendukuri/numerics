# CUDA — detection (Phase 1) and target configuration (Phase 2)

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

# Phase 2: Target configuration
if(TARGET numerics_raw_kernel AND NUMERICS_HAS_CUDA)
    target_link_libraries(numerics_raw_kernel INTERFACE CUDA::cudart)
    target_compile_definitions(numerics_raw_kernel INTERFACE NUMERICS_HAS_CUDA)
    set_target_properties(numerics PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE  ON
    )
endif()
