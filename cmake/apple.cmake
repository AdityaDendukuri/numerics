# Apple platform configurations
if(NOT APPLE)
    return()
endif()

if(POLICY CMP0167)
    cmake_policy(SET CMP0167 NEW)
endif()

target_compile_definitions(numerics_raw_kernel INTERFACE ACCELERATE_NEW_LAPACK)

# Accelerate keeps legacy CBLAS entry points for source compatibility but
# marks them deprecated even when they are the supported ABI on this platform.
target_compile_options(numerics_raw_kernel INTERFACE
    $<$<COMPILE_LANG_AND_ID:CXX,AppleClang,Clang>:-Wno-deprecated-declarations>)
