# FFTW3 -- spectral transforms backend
#
# Searches for a system FFTW3 installation via pkg-config then find_library.
# Falls back gracefully (fftw -> seq) if absent.
#
# Delegates to cmake/FindFFTW3.cmake which creates a stable FFTW3::FFTW3
# imported target.  That target is serialisable in NumericsTargets.cmake and
# re-findable by consumers via find_dependency(FFTW3).
#
# Sets: NUMERICS_HAS_FFTW compile definition on the numerics target.
#
# Install:
#   macOS:  brew install fftw
#   Ubuntu: apt install libfftw3-dev

if(NOT NUMERICS_USE_FFTW)
    return()
endif()

# FindFFTW3.cmake lives alongside this file.
list(PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}")
find_package(FFTW3 QUIET)

if(NOT FFTW3_FOUND)
    message(STATUS "FFTW3: not found -- spectral backend falls back to seq")
    message(STATUS "       Install: brew install fftw  |  apt install libfftw3-dev")
    return()
endif()

# fftw3.h is only included in .cpp files; consumers do not need the header path.
target_link_libraries(numerics PUBLIC FFTW3::FFTW3)
target_compile_definitions(numerics PUBLIC NUMERICS_HAS_FFTW)
message(STATUS "FFTW3: found  (${FFTW3_LIBRARY})")
