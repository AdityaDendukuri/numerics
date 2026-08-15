# FFTW3 -- spectral transforms backend

if(NOT NUMERICS_USE_FFTW)
    return()
endif()

list(PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}")
find_package(FFTW3 QUIET)

if(NOT FFTW3_FOUND)
    message(STATUS "FFTW3: not found -- spectral backend falls back to seq")
    message(STATUS "       Install: brew install fftw  |  apt install libfftw3-dev")
    return()
endif()

target_link_libraries(numerics_raw_kernel INTERFACE FFTW3::FFTW3)
target_compile_definitions(numerics_raw_kernel INTERFACE NUMERICS_HAS_FFTW)
message(STATUS "FFTW3: found  (${FFTW3_LIBRARY})")
