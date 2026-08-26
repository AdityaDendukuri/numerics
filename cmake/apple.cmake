# Apple platform configurations
if(NOT APPLE)
    return()
endif()

if(POLICY CMP0167)
    cmake_policy(SET CMP0167 NEW)
endif()

if(TARGET numerics_backend_blas)
    target_compile_definitions(numerics_backend_blas INTERFACE ACCELERATE_NEW_LAPACK)
endif()
if(TARGET numerics_backend_lapack)
    target_compile_definitions(numerics_backend_lapack INTERFACE ACCELERATE_NEW_LAPACK)
endif()

# Accelerate keeps legacy CBLAS entry points for source compatibility but marks
# them deprecated even when they are the supported ABI on this platform.
#
# This suppression applies to numerics' own translation units only. It was
# previously INTERFACE on numerics_kernel, which every target links PUBLIC, so
# it reached consumers and silently disabled every deprecation diagnostic in their
# builds as well.
foreach(_num_tgt numerics_precompiled numerics_mpi numerics_cuda)
    if(TARGET ${_num_tgt})
        get_target_property(_num_type ${_num_tgt} TYPE)
        if(NOT _num_type STREQUAL "INTERFACE_LIBRARY")
            target_compile_options(${_num_tgt} PRIVATE
                $<$<COMPILE_LANG_AND_ID:CXX,AppleClang,Clang>:-Wno-deprecated-declarations>)
        endif()
    endif()
endforeach()
