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

# Phase 2: Device target configuration. CUDA is an opt-in compiled capability;
# it never changes the kernel, core, or host-container target interfaces.
# The .cu sources live in numerics_cuda,
# so the device build properties belong there and this has to wait until that
# target has been created.
if(TARGET numerics_cuda)
    target_link_libraries(numerics_cuda PUBLIC CUDA::cudart)
    target_compile_definitions(numerics_cuda PUBLIC NUMERICS_HAS_CUDA)
    set_target_properties(numerics_cuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE  ON
    )
endif()
